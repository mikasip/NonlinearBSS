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
#' The scaled MSHAP values divide the values of each variable by their sum, so
#' that they give the relative contributions of the components to the
#' variable and sum to one over the components. The components are ordered by
#' their average scaled MSHAP over the variables if `reorder = TRUE`. This is
#' the approach used to interpret the latent components of air pollution and
#' weather data by \insertCite{sipila2026interpretable;textual}{NonlinearBSS}.
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
#' @param ... Further graphical parameters passed to [graphics::image()].
#' @return `x`, invisibly.
#' @export
plot.mshap <- function(x, scaled = TRUE, values = TRUE, digits = 2,
                       col = hcl.colors(50, "YlOrRd", rev = TRUE), ...) {
  M <- if (scaled) x$scaled else x$mshap
  p <- nrow(M)
  L <- ncol(M)
  label_lines <- max(graphics::strwidth(rownames(M), units = "inches")) /
    par("csi")
  op <- par(mar = c(4, max(4, label_lines + 1.5), 2, 1))
  on.exit(par(op))
  image(seq_len(L), seq_len(p), t(M[p:1, , drop = FALSE]), col = col,
        axes = FALSE, xlab = "", ylab = "",
        main = if (scaled) "Scaled MSHAP" else "MSHAP", ...)
  axis(1, at = seq_len(L), labels = colnames(M), las = 2, tick = FALSE)
  axis(2, at = seq_len(p), labels = rev(rownames(M)), las = 1, tick = FALSE)
  if (values) {
    graphics::text(rep(seq_len(L), each = p), rep(p:1, L),
                   format(round(as.vector(M), digits), nsmall = digits),
                   cex = 0.8,
                   col = ifelse(as.vector(M) > mean(range(M)), "white", "black"))
  }
  box()
  invisible(x)
}
