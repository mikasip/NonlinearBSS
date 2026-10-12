#' Selecting the Number of Latent Components
#'
#' Compares models with different numbers of latent components by their
#' evidence lower bound (ELBO) and the profile AIC (pAIC).
#'
#' @details
#' For a fitted model with \eqn{L} latent components, the profile AIC is
#' \deqn{\mathrm{pAIC} = \mathrm{ELBO} - L,}{pAIC = ELBO - L,}
#' where the ELBO is the average evidence lower bound per observation (the
#' `elbo` element of the fitted object). Like the Akaike information
#' criterion, it rewards fit and penalises the number of components. The
#' number of components with the largest pAIC is selected
#' \insertCite{sipila2024modelling,sipila2026interpretable}{NonlinearBSS}.
#'
#' The models should differ only in the number of latent components: the
#' ELBO depends also on the data, the auxiliary variables and the
#' reconstruction error scale `error_dist_sigma`, and the values are not
#' comparable otherwise. `latent_dim_selection()` warns if the models differ
#' in these respects.
#'
#' `latent_dim_selection()` compares models that have already been fitted,
#' which is useful when the models are fitted separately, for example in
#' different sessions. `select_latent_dim()` fits the models for the given
#' numbers of components and then compares them. Using the same `seed` for
#' all models (the default when `seed` is given) makes the comparison less
#' affected by random variation in the training.
#'
#' @param object A model fitted with [iVAE()], [iVAEar()] or [VAE()], or a
#'   list of such models.
#' @return `pAIC()` returns the pAIC values. `latent_dim_selection()` and
#'   `select_latent_dim()` return an object of class `latent_dim_selection`,
#'   a list with elements
#'   \item{table}{A data frame with the number of latent components, the
#'   ELBO and the pAIC of each model, ordered by the number of components.}
#'   \item{selected}{The selected number of latent components.}
#'   \item{models}{The fitted models, in the order of `table`, or `NULL` if
#'   `keep_models = FALSE`.}
#' @seealso [plot.latent_dim_selection()]
#' @references \insertAllCited{}
#' @examplesIf tf_available()
#' set.seed(1)
#' src <- simulate_sources(2, time = 1:500, n_segments_time = 5)
#' x <- mix_data(src$sources, n_layers = 2, output_dim = 4)
#'
#' # fit the models with the wrapper (few epochs for speed: use more in
#' # practice, as undertrained models can select the wrong number) ...
#' sel <- select_latent_dim(x, latent_dims = 1:3, aux = aux_segments(100),
#'                          time = 1:500, epochs = 3, verbose = 0, seed = 1)
#' sel
#' plot(sel)
#' best <- sel$models[[which(sel$table$latent_dim == sel$selected)]]
#'
#' # ... or compare models fitted separately
#' fits <- lapply(1:2, function(L) {
#'   iVAE(x, aux_segments(100), latent_dim = L, time = 1:500, epochs = 3,
#'        verbose = 0, seed = 1)
#' })
#' pAIC(fits)
#' latent_dim_selection(fits)
#' @export
pAIC <- function(object) {
  if (inherits(object, "nlbss")) {
    check_elbo(object)
    return(object$elbo - object$latent_dim)
  }
  if (!is.list(object)) {
    stop("`object` must be a fitted model or a list of fitted models.",
         call. = FALSE)
  }
  vapply(object, pAIC, numeric(1))
}

check_elbo <- function(object) {
  if (!inherits(object, "nlbss") || is.null(object$elbo)) {
    stop("The models must be fitted with iVAE(), iVAEar() or VAE(); ",
         "TCL models have no ELBO.", call. = FALSE)
  }
}

#' @rdname pAIC
#' @param keep_models Logical. Keep the fitted models in the result?
#' @export
latent_dim_selection <- function(object, keep_models = TRUE) {
  if (inherits(object, "nlbss")) object <- list(object)
  if (!is.list(object) || length(object) == 0) {
    stop("`object` must be a list of fitted models.", call. = FALSE)
  }
  lapply(object, check_elbo)
  L <- vapply(object, function(m) m$latent_dim, numeric(1))
  if (anyDuplicated(L)) {
    warning("Several models have the same number of latent components.",
            call. = FALSE)
  }
  same <- function(f) length(unique(vapply(object, function(m) {
    paste(deparse(f(m)), collapse = "")
  }, ""))) == 1
  if (!same(function(m) m$method) || !same(function(m) m$n) ||
      !same(function(m) m$var_names) ||
      !same(function(m) m$spec$error_dist_sigma) ||
      !same(function(m) m$spec$error_dist) ||
      !same(function(m) vapply(m$aux$specs, format, ""))) {
    warning("The models differ in more than the number of latent ",
            "components (method, data, auxiliary data or error ",
            "distribution); their ELBO values may not be comparable.",
            call. = FALSE)
  }
  o <- order(L)
  tab <- data.frame(latent_dim = L[o],
                    elbo = vapply(object, function(m) m$elbo, numeric(1))[o],
                    pAIC = pAIC(object)[o])
  structure(list(table = tab,
                 selected = tab$latent_dim[which.max(tab$pAIC)],
                 models = if (keep_models) object[o]),
            class = "latent_dim_selection")
}

#' @rdname pAIC
#' @param x The data, in any format accepted by `method`.
#' @param latent_dims The numbers of latent components to compare.
#' @param method The fitting function: [iVAE()], [iVAEar()] or [VAE()].
#' @param ... Further arguments passed to `method`, for example `aux`,
#'   `coords`, `time`, `epochs` and `seed`.
#' @param progress Logical. Report the progress?
#' @export
select_latent_dim <- function(x, latent_dims, method = iVAE, ...,
                              keep_models = TRUE, progress = interactive()) {
  if (!any(vapply(list(iVAE, iVAEar, VAE), identical, logical(1), method))) {
    stop("`method` must be iVAE, iVAEar or VAE.", call. = FALSE)
  }
  latent_dims <- sort(unique(as.integer(latent_dims)))
  data_name <- deparse1(substitute(x))
  fits <- lapply(latent_dims, function(L) {
    if (progress) {
      message("Fitting a model with ", L, " latent component",
              if (L != 1) "s")
    }
    fit <- method(x, latent_dim = L, ...)
    fit$data_name <- data_name
    fit
  })
  latent_dim_selection(fits, keep_models = keep_models)
}

#' @export
print.latent_dim_selection <- function(x, digits = 4, ...) {
  cat("Selection of the number of latent components\n\n")
  tab <- x$table
  tab$selected <- ifelse(tab$latent_dim == x$selected, "*", "")
  print(format(tab, digits = digits), row.names = FALSE)
  cat("\nSelected by the largest pAIC = ELBO - number of components:",
      x$selected, "\n")
  invisible(x)
}

#' Plot the Selection of the Number of Latent Components
#'
#' Plots the ELBO (or pAIC) of models with different numbers of latent
#' components and highlights the selected number.
#'
#' @param x An object returned by [latent_dim_selection()] or
#'   [select_latent_dim()].
#' @param which The criterion to plot, `"elbo"` or `"pAIC"`.
#' @param ... Further graphical parameters passed to [plot()].
#' @return `x`, invisibly.
#' @export
plot.latent_dim_selection <- function(x, which = c("elbo", "pAIC"), ...) {
  which <- match.arg(which)
  tab <- x$table
  y <- tab[[which]]
  plot(tab$latent_dim, y, type = "b", pch = 19, xaxt = "n",
       xlab = "Number of latent components",
       ylab = if (which == "elbo") "ELBO" else "pAIC", ...)
  axis(1, at = tab$latent_dim)
  sel <- tab$latent_dim == x$selected
  graphics::points(tab$latent_dim[sel], y[sel], col = "red", pch = 19, cex = 1.8)
  invisible(x)
}
