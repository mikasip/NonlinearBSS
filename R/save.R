#' Save and Load Fitted Models
#'
#' Saves a fitted model to a file and loads it back, restoring the trained
#' TensorFlow models.
#'
#' The Keras models of a fitted object live in the Python session and are
#' lost when R is restarted. The fitted object therefore also stores the
#' trained network weights, and the models are rebuilt automatically when
#' needed. `save_nlbss()` saves the object without the Python references, and
#' `load_nlbss()` restores the models right away. Saving with [saveRDS()] and
#' loading with [readRDS()] works as well; the models are then rebuilt on
#' first use.
#'
#' @param object A fitted model from [iVAE()], [iVAEar()], [VAE()] or [TCL()].
#' @param file Path of the file.
#' @return `save_nlbss()` returns the path invisibly, `load_nlbss()` the
#'   fitted model.
#' @examplesIf tf_available()
#' x <- mix_data(matrix(rnorm(3000), ncol = 3), n_layers = 2)
#' fit <- VAE(x, epochs = 2, verbose = 0)
#' file <- tempfile(fileext = ".rds")
#' save_nlbss(fit, file)
#' fit2 <- load_nlbss(file)
#' all.equal(predict(fit, x[1:5, ]), predict(fit2, x[1:5, ]))
#' @export
save_nlbss <- function(object, file) {
  if (!inherits(object, "nlbss")) stop("`object` must be a fitted nlbss model.", call. = FALSE)
  object$models <- new.env(parent = emptyenv())
  saveRDS(object, file)
  invisible(file)
}

#' @rdname save_nlbss
#' @export
load_nlbss <- function(file) {
  object <- readRDS(file)
  if (!inherits(object, "nlbss")) stop("The file does not contain a fitted nlbss model.", call. = FALSE)
  if (!is.environment(object$models)) object$models <- new.env(parent = emptyenv())
  get_net(object)
  object
}
