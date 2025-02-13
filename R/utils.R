norm_log_pdf <- function(x, mu, v, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    lpdf <- tf$constant(-0.5, "float32") * (tf$math$pow((x - mu), 2) / v + tf$cast(tf$math$log(2 * pi), "float32") + tf$math$log(v))
    if (reduce) {
        return(tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

huber_loss <- function(x, mu, v, delta = 0.2, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    abs_diff <- tf$abs(x - mu)
    delta <- tf$constant(delta, "float32")
    quadratic_part <- tf$constant(-1/2, "float32") * tf$square(x - mu)
    linear_part <- -delta * (abs_diff - tf$constant(0.5, "float32") * delta)
    huber_loss <- tf$where(abs_diff < delta, quadratic_part, linear_part)
    lpdf <- huber_loss / v # + tf$constant(-0.5, "float32") * (tf$cast(tf$math$log(2 * pi), "float32") + tf$math$log(v))
    if (reduce) {
        return(tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

laplace_log_pdf <- function(x, loc, scale, reduce = TRUE) {
    scale <- scale + 1e-35
    lpdf <- -(tf$math$log(2 * scale) + tf$abs(x - loc) / (scale))
    if (reduce) {
        return(tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

# sigma not used
bernoulli_log_pdf <- function(x, theta, sigma, reduce = TRUE) {
    theta <- theta * 0.99 + 0.005
    lpdf <- x * tf$math$log(theta) + (1 - x) * tf$math$log(1 - theta)
    if (reduce) {
        return(tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

absolute_activation <- function(x) {
    tf$abs(x)
}

SamplingGaussian(keras$layers$Layer) %py_class% {
    initialize <- function(p) {
        super$initialize()
        self$p <- p
    }

    call <- function(y) {
        z_mean <- y[, 1:self$p]
        z_log_var <- y[, (self$p + 1):(2 * self$p)]
        epsilon <- tf$random$normal(shape(self$p))
        z_mean + tf$math$exp(z_log_var / 2) * epsilon
    }
}

sampling_gaussian <- create_layer_wrapper(SamplingGaussian)

SamplingLaplace(keras$layers$Layer) %py_class% {
    initialize <- function(p) {
        super$initialize()
        self$p <- p
    }

    call <- function(y) {
        z_mean <- y[, 1:self$p]
        log_scale <- y[, (self$p + 1):(2 * self$p)]
        cum_vals <- tf$random$uniform(shape(self$p))
        epsilon <- tf$sign(cum_vals - 0.5) * log(2 * tf$abs(cum_vals - 0.5))
        z_mean - tf$math$exp(log_scale) * epsilon
    }
}

sampling_laplace <- create_layer_wrapper(SamplingLaplace)



increase_batch_callback <- CustomCallback(keras$callbacks$Callback) %py_class% {
  
    initialize <- function(epoch_threshold, batch_sizes, inputs, 
        outputs, validation_data, total_epochs) {
      super$initialize()
      self$epoch_threshold <- epoch_threshold
      self$batch_sizes <- batch_sizes
      self$inputs <- inputs
      self$outputs <- outputs
      self$validation_data <- validation_data
      self$start_epoch <- as.integer(0)
      self$total_epochs <- total_epochs
    }
    
    on_epoch_begin <- function(epoch, logs = NULL) {
      if ((epoch + self$start_epoch) %in% self$epoch_threshold) {
        new_batch_size <- self$batch_sizes[which(self$epoch_threshold == epoch + self$start_epoch)]
        cat(sprintf("\nIncreasing batch size to %d at epoch %d\n", new_batch_size, epoch + self$start_epoch))
        self$start_epoch <- as.integer(epoch + self$start_epoch)

        if (self$start_epoch >= self$total_epochs) {
            cat("\nTraining completed at epoch", epoch + self$start_epoch, "\n")
            self$model$stop_training <- TRUE  # Ensure stopping if final epoch is reached
            return()
        }
        # Stop training and restart with new batch size
        self$model$stop_training <- TRUE
        self$model$fit(
          self$inputs, self$outputs,
          batch_size = as.integer(new_batch_size),
          epochs = as.integer(self$params$epochs - epoch),
          validation_data = self$validation_data,
          callbacks = list(self)
        )
      }
    }
}

