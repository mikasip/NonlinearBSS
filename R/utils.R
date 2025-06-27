norm_log_pdf <- function(x, mu, v, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    lpdf <- tensorflow::tf$constant(-0.5, "float32") * (tensorflow::tf$math$pow((x - mu), 2) / v + tensorflow::tf$cast(tensorflow::tf$math$log(2 * pi), "float32") + tensorflow::tf$math$log(v))
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

huber_loss <- function(x, mu, v, delta = 0.2, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    abs_diff <- tensorflow::tf$abs(x - mu)
    delta <- tensorflow::tf$constant(delta, "float32")
    quadratic_part <- tensorflow::tf$constant(-1/2, "float32") * tensorflow::tf$square(x - mu)
    linear_part <- -delta * (abs_diff - tensorflow::tf$constant(0.5, "float32") * delta)
    huber_loss <- tensorflow::tf$where(abs_diff < delta, quadratic_part, linear_part)
    lpdf <- huber_loss / v # + tensorflow::tf$constant(-0.5, "float32") * (tensorflow::tf$cast(tensorflow::tf$math$log(2 * pi), "float32") + tensorflow::tf$math$log(v))
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

laplace_log_pdf <- function(x, loc, scale, reduce = TRUE) {
    scale <- scale + 1e-35
    lpdf <- -(tensorflow::tf$math$log(2 * scale) + tensorflow::tf$abs(x - loc) / (scale))
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

# sigma not used
bernoulli_log_pdf <- function(x, theta, sigma, reduce = TRUE) {
    theta <- theta * 0.99 + 0.005
    lpdf <- x * tensorflow::tf$math$log(theta) + (1 - x) * tensorflow::tf$math$log(1 - theta)
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

absolute_activation <- keras3::Layer(
    "AbsoluteActivation",
    initialize = function() {
        super$initialize()
    },
    call = function(y) {
        tensorflow::tf$abs(y)
    }
)

sampling_gaussian <- keras3::Layer(
    "SamplingGaussian",
    initialize = function(p) {
        super$initialize()
        self$p <- p
    },
    call = function(y) {
        z_mean <- y[, 1:self$p]
        z_log_var <- y[, (self$p + 1):(2 * self$p)]
        epsilon <- tensorflow::tf$random$normal(shape(self$p))
        z_mean + tensorflow::tf$math$exp(z_log_var / 2) * epsilon
    }
)

sampling_laplace <- keras3::Layer(
    "SamplingLaplace",
    initialize = function(p) {
        super$initialize()
        self$p <- p
    },
    call = function(y) {
        z_mean <- y[, 1:self$p]
        log_scale <- y[, (self$p + 1):(2 * self$p)]
        cum_vals <- tensorflow::tf$random$uniform(shape(self$p))
        epsilon <- tensorflow::tf$sign(cum_vals - 0.5) * log(2 * tensorflow::tf$abs(cum_vals - 0.5))
        z_mean - tensorflow::tf$math$exp(log_scale) * epsilon
    }
)

find_unique_name <- function(name, i = 0) {
    if (dir.exists(name)) {
        find_unique_name(paste0(name, i + 1))
    } else {
        return(name)
    }
}
