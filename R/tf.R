#' Check Whether TensorFlow Is Available
#'
#' Checks that Python, TensorFlow and Keras can be used by \pkg{keras3}. The
#' model fitting functions of the package need a working TensorFlow
#' installation.
#'
#' Usually nothing has to be installed by hand: on first use, \pkg{reticulate}
#' downloads a suitable Python version together with TensorFlow and Keras
#' into a cached environment. On Windows, TensorFlow needs a Python version
#' below 3.13, which the package requests automatically. If you configure
#' Python yourself (for example with `RETICULATE_PYTHON` or
#' `reticulate::use_virtualenv()`), the environment must contain the Python
#' packages `tensorflow` and `keras`.
#'
#' Checking for TensorFlow requires initialising Python. To avoid triggering
#' an automatic Python installation in non-interactive sessions (for example
#' during `R CMD check`), Python is only initialised if `initialize = TRUE`.
#' When Python has already been initialised, the check is always done.
#'
#' @param initialize Logical. Is Python allowed to be initialised by the
#'   check? Defaults to `TRUE` in interactive sessions and in package checks
#'   with the environment variable `NOT_CRAN` set to `"true"`.
#' @return `TRUE` if TensorFlow and Keras can be imported, `FALSE` otherwise.
#' @examples
#' tf_available()
#' @export
tf_available <- function(initialize = interactive() ||
                           identical(Sys.getenv("NOT_CRAN"), "true")) {
  if (!initialize && !reticulate::py_available(initialize = FALSE)) {
    return(FALSE)
  }
  if (!reticulate::py_available(initialize = FALSE)) declare_python()
  tryCatch(
    reticulate::py_module_available("tensorflow") &&
      reticulate::py_module_available("keras"),
    error = function(e) FALSE
  )
}

check_tf <- function() {
  if (!tf_available(initialize = TRUE)) {
    stop("TensorFlow could not be loaded. Restart R and try again: Python ",
         "and TensorFlow are then installed automatically. If you set up ",
         "Python yourself, make sure it has the packages 'tensorflow' and ",
         "'keras'. See ?tf_available.", call. = FALSE)
  }
  invisible(TRUE)
}

# Declare the Python requirements before Python is initialised. Loading
# keras3 declares Keras and TensorFlow. On Windows the tensorflow R package
# also requires numpy < 2, for which TensorFlow has no wheels for Python
# 3.13 or newer, so a compatible Python version is requested.
declare_python <- function() {
  loadNamespace("keras3")
  if (.Platform$OS.type == "windows") {
    tryCatch(reticulate::py_require(python_version = ">=3.9,<3.13"),
             error = function(e) NULL)
  }
  invisible()
}

# Keras layer classes are created lazily, once per session, so that loading
# the package never initialises Python.
.layers <- new.env(parent = emptyenv())

nlbss_layer <- function(name) {
  if (is.null(.layers[[name]])) {
    .layers[[name]] <- switch(name,
      sampling = make_sampling_layer(),
      elbo = make_elbo_layer()
    )
  }
  .layers[[name]]
}

make_sampling_layer <- function() {
  keras3::Layer(
    "NLBSSSampling",
    initialize = function(dist = "gaussian", seed = NULL, ...) {
      super$initialize(...)
      self$dist <- dist
      self$seed_generator <- keras3::random_seed_generator(seed)
    },
    call = function(inputs) {
      z_mean <- inputs[[1]]
      z_log_scale <- inputs[[2]]
      shape <- keras3::op_shape(z_mean)
      if (self$dist == "gaussian") {
        eps <- keras3::random_normal(shape, seed = self$seed_generator)
        z_mean + keras3::op_exp(z_log_scale / 2) * eps
      } else {
        u <- keras3::random_uniform(shape, minval = -0.4999, maxval = 0.4999,
                                    seed = self$seed_generator)
        eps <- -keras3::op_sign(u) * keras3::op_log(1 - 2 * keras3::op_abs(u))
        z_mean + keras3::op_exp(z_log_scale) * eps
      }
    }
  )
}

# Log densities used in the loss. For "gaussian" the scale is a variance,
# for "laplace" it is the Laplace scale parameter.
tf_log_density <- function(dist, x, loc, scale) {
  switch(dist,
    gaussian = -0.5 * (keras3::op_square(x - loc) / scale + log(2 * pi) +
                         keras3::op_log(scale)),
    laplace = -(keras3::op_log(2 * scale) + keras3::op_abs(x - loc) / scale),
    huber = {
      d <- keras3::op_abs(x - loc)
      keras3::op_where(d < 0.2, -0.5 * keras3::op_square(x - loc),
                       -0.2 * (d - 0.1)) / scale
    },
    poisson = x * loc - keras3::op_exp(loc),
    bernoulli = {
      theta <- loc * 0.99 + 0.005
      x * keras3::op_log(theta) + (1 - x) * keras3::op_log(1 - theta)
    }
  )
}

make_elbo_layer <- function() {
  keras3::Layer(
    "NLBSSElbo",
    initialize = function(error_dist, source_dist, sigma_start, sigma_end,
                          steps, step, ...) {
      super$initialize(...)
      self$error_dist <- error_dist
      self$source_dist <- source_dist
      self$sigma_start <- sigma_start
      self$sigma_end <- sigma_end
      self$steps <- max(steps, 1)
      self$step <- step
    },
    call = function(inputs) {
      x <- inputs[[1]]
      x_hat <- inputs[[2]]
      mask <- inputs[[3]]
      z <- inputs[[4]]
      z_mean <- inputs[[5]]
      z_log_scale <- inputs[[6]]
      prior_mean <- inputs[[7]]
      prior_log_scale <- inputs[[8]]
      progress <- keras3::op_clip(
        keras3::op_cast(self$step, "float32") / self$steps, 0, 1)
      sigma <- self$sigma_start + (self$sigma_end - self$sigma_start) * progress
      log_px <- tf_log_density(self$error_dist, x, x_hat, sigma) * mask
      log_qz <- tf_log_density(self$source_dist, z, z_mean,
                               keras3::op_exp(z_log_scale))
      log_pz <- tf_log_density(self$source_dist, z, prior_mean,
                               keras3::op_exp(prior_log_scale))
      keras3::op_concatenate(list(
        keras3::op_sum(log_px, axis = -1L, keepdims = TRUE),
        keras3::op_sum(log_pz - log_qz, axis = -1L, keepdims = TRUE)
      ), axis = -1L)
    }
  )
}

# The training model outputs per observation (log p(x|z), log p(z|u) - log q(z|x,u)).
elbo_loss <- function(y_true, y_pred) {
  -keras3::op_mean(y_pred[, 1] + y_pred[, 2])
}

elbo_metrics <- function() {
  list(
    keras3::custom_metric("reconstruction", function(y_true, y_pred) {
      keras3::op_mean(y_pred[, 1])
    }),
    keras3::custom_metric("kl", function(y_true, y_pred) {
      -keras3::op_mean(y_pred[, 2])
    })
  )
}

# R versions of the log densities, used to evaluate the final ELBO.
r_log_density <- function(dist, x, loc, scale) {
  switch(dist,
    gaussian = -0.5 * ((x - loc)^2 / scale + log(2 * pi) + log(scale)),
    laplace = -(log(2 * scale) + abs(x - loc) / scale),
    huber = {
      d <- abs(x - loc)
      ifelse(d < 0.2, -0.5 * (x - loc)^2, -0.2 * (d - 0.1)) / scale
    },
    poisson = x * loc - exp(loc),
    bernoulli = {
      theta <- loc * 0.99 + 0.005
      x * log(theta) + (1 - x) * log(1 - theta)
    }
  )
}

default_optimizer <- function(lr_start, lr_end, steps) {
  keras3::optimizer_adam(
    learning_rate = keras3::learning_rate_schedule_polynomial_decay(
      initial_learning_rate = lr_start, decay_steps = as.integer(steps),
      end_learning_rate = lr_end, power = 2
    )
  )
}

predict_np <- function(model, inputs, batch_size = 4096L) {
  out <- stats::predict(model, inputs, batch_size = as.integer(batch_size),
                        verbose = 0)
  if (is.list(out)) lapply(out, as.matrix) else as.matrix(out)
}
