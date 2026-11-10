#' Mean Absolute SHAP Values of the Latent Components
#'
#' Measures how much each latent component contributes to each observed
#' variable, using kernel SHAP values \insertCite{kernelshap}{NonlinearBSS}
#' of the decoder, the estimated mixing function.
#'
#' @details
#' The decoder maps the latent components to the observed variables. For each
#' observed variable, kernel SHAP \insertCite{kernelshap}{NonlinearBSS}
#' decomposes the decoder output at an observation into contributions of the
#' latent components, relative to a background sample. The mean absolute SHAP
#' value (MSHAP) of component \eqn{j} for variable \eqn{k} is the average of
#' the absolute contributions over the explained observations, see
#' \insertCite{marcilio2020explanations;textual}{NonlinearBSS} for SHAP based
#' variable importance.
#'
#' The scaled MSHAP values \insertCite{sipila2024spatial}{NonlinearBSS} 
#' divide the values of each variable by their sum, so
#' that they give the relative contributions of the components to the
#' variable and sum to one over the components. The components are ordered by
#' their average scaled MSHAP over the variables if `reorder = TRUE`.
#'
#' Kernel SHAP is computationally heavy, so the values are computed for a
#' random subsample of `n_sample` observations. By default, the background
#' data are the first `n_background` observations of this subsample. Larger
#' `n_sample` and `n_background` give more accurate values. With more than 8
#' components the SHAP values are approximated, see
#' [kernelshap::kernelshap()].
#'
#' @param object A model fitted with [iVAE()], [iVAEar()] or [VAE()].
#' @param S Optional matrix of latent components (on the standardised scale
#'   of the `S` element of `object`) to explain. Defaults to the components of
#'   the training data.
#' @param n_sample Number of observations sampled from `S` to explain.
#' @param n_background Number of observations of the sample used as
#'   background data. Ignored if `background` is given.
#' @param background Optional matrix of components used as background data.
#' @param reorder Logical. Order the components by their average scaled
#'   MSHAP?
#' @param seed Optional random seed for the subsampling and kernel SHAP.
#' @param verbose Logical. Show the progress of kernel SHAP?
#' @param ... Further arguments passed to [kernelshap::kernelshap()].
#' @return An object of class `mshap`, a list with elements
#'   \item{mshap}{Matrix of mean absolute SHAP values, one row per observed
#'   variable and one column per component.}
#'   \item{scaled}{The values scaled to sum to one for each variable.}
#'   \item{order}{The order of the components by average scaled MSHAP. The
#'   columns of `mshap` and `scaled` are in this order if `reorder = TRUE`.}
#'   \item{shap}{The SHAP values: a list with one matrix per observed
#'   variable, one row per explained observation and one column per
#'   component.}
#'   \item{X, background}{The explained components and the background data.}
#'   \item{kernelshap}{The output of [kernelshap::kernelshap()].}
#' @seealso [decode()], [plot.mshap()]
#' @references \insertAllCited{}
#' @examplesIf tf_available()
#' set.seed(1)
#' src <- simulate_sources(3, time = 1:1000, n_segments_time = 10)
#' x <- mix_data(src$sources, n_layers = 2)
#' fit <- iVAE(x, aux_segments(100), time = 1:1000, epochs = 5, verbose = 0)
#' ms <- mshap(fit, n_sample = 100, n_background = 20, seed = 1)
#' ms
#' plot(ms)
#' @export
mshap <- function(object, S = NULL, n_sample = 1000, n_background = 100,
                  background = NULL, reorder = TRUE, seed = NULL,
                  verbose = FALSE, ...) {
  if (!inherits(object, "nlbss") || object$method == "TCL") {
    stop("`object` must be a model fitted by iVAE(), iVAEar() or VAE().",
         call. = FALSE)
  }
  if (is.null(S)) S <- object$S
  S <- as.matrix(S)
  if (ncol(S) != object$latent_dim) {
    stop("`S` must have ", object$latent_dim, " columns.", call. = FALSE)
  }
  colnames(S) <- colnames(object$S)
  if (!is.null(seed)) {
    old_rng <- set_seed(seed, tf = FALSE)
    on.exit(restore_rng(old_rng), add = TRUE)
  }
  if (n_sample < nrow(S)) S <- S[sample(nrow(S), n_sample), , drop = FALSE]
  if (is.null(background)) {
    background <- S[seq_len(min(n_background, nrow(S))), , drop = FALSE]
  } else {
    background <- as.matrix(background)
    colnames(background) <- colnames(S)
  }

  ks <- kernelshap::kernelshap(
    object, X = as.data.frame(S), bg_X = as.data.frame(background),
    pred_fun = function(object, X) decode(object, as.matrix(X)),
    verbose = verbose, seed = seed, ...
  )
  shap <- if (is.list(ks$S)) ks$S else list(ks$S)
  names(shap) <- object$var_names
  raw <- do.call(rbind, lapply(shap, function(s) colMeans(abs(s))))
  dimnames(raw) <- list(object$var_names, colnames(S))
  scaled <- raw / rowSums(raw)
  ord <- order(-colMeans(scaled))
  if (reorder) {
    raw <- raw[, ord, drop = FALSE]
    scaled <- scaled[, ord, drop = FALSE]
  }
  structure(list(mshap = raw, scaled = scaled, order = ord, shap = shap,
                 X = S, background = background, kernelshap = ks,
                 reordered = reorder),
            class = "mshap")
}

#' @export
print.mshap <- function(x, digits = 3, ...) {
  cat("Scaled mean absolute SHAP values (rows: variables, columns: components)\n\n")
  tab <- rbind(x$scaled, Average = colMeans(x$scaled))
  print(round(tab, digits))
  cat("\nBased on", nrow(x$X), "observations and", nrow(x$background),
      "background observations.\n")
  invisible(x)
}

#' Plot Mean Absolute SHAP Values
#'
#' Shows the (scaled) mean absolute SHAP values of [mshap()] as a heatmap with
#' the observed variables as rows and the latent components as columns.
#'
#' @param x An object returned by [mshap()].
#' @param scaled Logical. Plot the scaled values?
#' @param values Logical. Print the values in the cells?
#' @param digits Number of decimals of the printed values.
#' @param col Colour palette.
#' @param ... Further arguments passed to [ggplot2::geom_tile()].
#' @return A `ggplot` object.
#' @export
plot.mshap <- function(x, scaled = TRUE, values = TRUE, digits = 2,
                       col = hcl.colors(50, "YlOrRd", rev = TRUE), ...) {
  M <- if (scaled) x$scaled else x$mshap
  d <- data.frame(
    variable = factor(rep(rownames(M), ncol(M)), levels = rev(rownames(M))),
    component = factor(rep(colnames(M), each = nrow(M)), levels = colnames(M)),
    value = as.vector(M))
  p <- ggplot2::ggplot(d, ggplot2::aes(.data$component, .data$variable,
                                       fill = .data$value)) +
    ggplot2::geom_tile(colour = "white", ...) +
    ggplot2::scale_fill_gradientn(colours = col, name = NULL) +
    ggplot2::labs(x = NULL, y = NULL,
                  title = if (scaled) "Scaled MSHAP" else "MSHAP") +
    ggplot2::coord_fixed() +
    theme_nlbss() +
    ggplot2::theme(panel.grid = ggplot2::element_blank())
  if (values) {
    d$label <- format(round(d$value, digits), nsmall = digits)
    d$light <- d$value > mean(range(M))
    p <- p + ggplot2::geom_text(
      data = d, ggplot2::aes(label = .data$label, colour = .data$light),
      size = 3, show.legend = FALSE) +
      ggplot2::scale_colour_manual(values = c(`TRUE` = "white", `FALSE` = "black"))
  }
  p
}
