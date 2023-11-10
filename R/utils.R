norm_log_pdf <- function(x, mu, v, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    lpdf <- tf$constant(-0.5, "float32") * (tf$math$pow((x - mu), 2) / v + tf$cast(tf$math$log(2 * pi), "float32") + tf$math$log(v))
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

WeightedSumLayer(keras$layers$Layer) %py_class% {
    initialize <- function(p, k) {
        super$initialize()
        self$p <- p
        self$k <- k
    }

    call <- function(y, weights) {
        z_sum <- 0
        ind <- 1
        for (i in 1:self$k) {
            z_i <- y[, ind:(ind + self$p - 1)]
            weight_i <- weights[, i]
            z_sum <- z_sum + z_i * weight_i
            ind <- ind + self$p
        }
        return(z_sum)
    }
}

weighted_sum <- create_layer_wrapper(WeightedSumLayer)
