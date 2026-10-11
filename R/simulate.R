#' Simulate Nonstationary Latent Sources
#'
#' Simulates independent latent components whose mean and variance, and
#' optionally autoregressive coefficient, change over spatial and/or
#' temporal segments. Such sources are identifiable by [iVAE()], [iVAEar()]
#' and [TCL()].
#'
#' @details
#' Spatial segments are formed by choosing `n_segments_space` of the distinct
#' locations at random as centres and assigning every location to the
#' nearest centre. Temporal segments are formed in the same way from the
#' distinct time points. For spatio-temporal data each combination of a
#' spatial and a temporal segment is its own segment.
#'
#' In each segment, each component has its own mean drawn from
#' \eqn{U}(`mean_range`) and standard deviation drawn from
#' \eqn{U}(`sd_range`). If `ar_range` is given, each component of each series
#' (location) follows a nonstationary AR(1) process
#' \deqn{z(t) = \mu(t) + \gamma(t) (z(t - 1) - \mu(t - 1)) + \sigma(t) \epsilon(t)}{
#' z(t) = mu(t) + gamma(t) (z(t-1) - mu(t-1)) + sigma(t) eps(t)}
#' with segment-wise coefficients \eqn{\gamma} drawn from \eqn{U}(`ar_range`)
#' and standard normal \eqn{\epsilon}; otherwise the observations are
#' independent normal.
#'
#' @param p Number of components.
#' @param coords Optional matrix of spatial coordinates, one row per
#'   observation.
#' @param time Optional vector of time points, one per observation. At least
#'   one of `coords` and `time` is needed.
#' @param n_segments_space Number of spatial segments.
#' @param n_segments_time Number of temporal segments.
#' @param mean_range Range of the segment means.
#' @param sd_range Range of the segment standard deviations.
#' @param ar_range Optional range of the autoregressive coefficients.
#'   Requires `time`.
#' @return A list with elements `sources` (an n x p matrix), `labels`
#'   (segment of each observation), `coords` and `time`.
#' @seealso [mix_data()]
#' @examples
#' # time series with variance changing in 10 segments
#' src <- simulate_sources(2, time = 1:1000, n_segments_time = 10)
#' plot.ts(src$sources)
#'
#' # spatial data
#' coords <- matrix(runif(1000), ncol = 2)
#' src <- simulate_sources(3, coords = coords, n_segments_space = 5)
#'
#' # spatio-temporal data with autoregressive sources
#' coords <- matrix(runif(40), ncol = 2)[rep(1:20, 50), ]
#' src <- simulate_sources(3, coords = coords, time = rep(1:50, each = 20),
#'                         n_segments_space = 3, n_segments_time = 5,
#'                         ar_range = c(-0.9, 0.9))
#' @export
simulate_sources <- function(p, coords = NULL, time = NULL,
                             n_segments_space = 10, n_segments_time = 10,
                             mean_range = c(-5, 5), sd_range = c(0.1, 5),
                             ar_range = NULL) {
  if (is.null(coords) && is.null(time)) {
    stop("Give `coords`, `time` or both.", call. = FALSE)
  }
  if (!is.null(ar_range) && is.null(time)) {
    stop("Autoregressive sources need `time`.", call. = FALSE)
  }
  if (!is.null(coords)) coords <- as.matrix(coords)
  n <- if (!is.null(coords)) nrow(coords) else length(time)
  lab_space <- rep(1L, n)
  if (!is.null(coords)) {
    key <- do.call(paste, c(as.data.frame(coords), sep = "_"))
    first <- !duplicated(key)
    U <- coords[first, , drop = FALSE]
    centres <- U[sample(nrow(U), min(n_segments_space, nrow(U))), , drop = FALSE]
    d2 <- outer(rowSums(U^2), rowSums(centres^2), "+") - 2 * U %*% t(centres)
    lab_space <- max.col(-d2, ties.method = "first")[match(key, key[first])]
  }
  lab_time <- rep(1L, n)
  if (!is.null(time)) {
    ut <- sort(unique(time))
    centres <- sample(ut, min(n_segments_time, length(ut)))
    lab_time <- max.col(-abs(outer(ut, centres, "-")),
                        ties.method = "first")[match(time, ut)]
  }
  labels <- as.integer(factor(paste(lab_space, lab_time)))
  k <- max(labels)
  draw <- function(range) matrix(runif(k * p, range[1], range[2]), k, p)
  mu <- draw(mean_range)[labels, , drop = FALSE]
  sigma <- draw(sd_range)[labels, , drop = FALSE]
  eps <- matrix(rnorm(n * p), n, p)
  if (is.null(ar_range)) {
    z <- mu + sigma * eps
  } else {
    gamma <- draw(ar_range)[labels, , drop = FALSE]
    series <- if (is.null(coords)) rep("1", n) else
      do.call(paste, c(as.data.frame(coords), sep = "_"))
    lag <- lag_index(time, series, 1)[, 1]
    z <- matrix(NA_real_, n, p)
    rank <- match(time, sort(unique(time)))
    for (r in sort(unique(rank))) {
      i <- which(rank == r)
      j <- lag[i]
      ok <- !is.na(j)
      zi <- mu[i, , drop = FALSE] + sigma[i, , drop = FALSE] * eps[i, , drop = FALSE]
      zi[ok, ] <- zi[ok, ] + gamma[i[ok], , drop = FALSE] *
        (z[j[ok], , drop = FALSE] - mu[j[ok], , drop = FALSE])
      z[i, ] <- zi
    }
  }
  colnames(z) <- paste0("Z", seq_len(p))
  list(sources = z, labels = labels, coords = coords, time = time)
}

#' Nonlinear Mixing of Data
#'
#' Mixes data with a random injective mixing function formed by a
#' multilayer perceptron. Useful for simulation studies.
#'
#' @details
#' The mixing function \eqn{f_L} with \eqn{L} layers is
#' \deqn{f_1(z) = B_1 z,\quad f_l(z) = B_l\, \omega(f_{l-1}(z)),\ l = 2, \dots, L,}{
#' f_1(z) = B_1 z,  f_l(z) = B_l omega(f_(l-1)(z)),}
#' where \eqn{B_l} are random matrices with orthonormal columns (obtained from
#' the singular value decomposition of matrices with \eqn{U(-1, 1)} entries)
#' and \eqn{\omega} is an elementwise nonlinearity. With one layer the mixing
#' is linear. After each layer the data are divided by the mean standard
#' deviation of the variables if `normalize = TRUE`. The first matrix maps
#' the data to `output_dim` dimensions; `output_dim` larger than the number
#' of columns of `x` gives overdetermined mixtures.
#'
#' The available nonlinearities are `"elu"` (\eqn{x} for \eqn{x \ge 0},
#' \eqn{e^x - 1} otherwise), `"lrelu"` (leaky ReLU with slope 0.2),
#' `"xtanh"` (\eqn{\tanh(x) + 0.1x}), `"smooth_lrelu"`, `"exp_arctan"`,
#' `"leaky_softplus"` and `"logsumexp"`.
#'
#' `final_activation = "exp"` or `"softplus"` makes the output positive, for
#' example to be used as Poisson rates.
#'
#' @param x A numeric matrix of latent components, one column per component.
#' @param n_layers Number of mixing layers.
#' @param nonlinearity The nonlinearity between layers.
#' @param output_dim Dimension of the mixed data, at least `ncol(x)`.
#' @param final_activation Activation applied to the output: `"none"`,
#'   `"exp"` or `"softplus"`.
#' @param normalize Logical. Scale the data after each layer?
#' @return A numeric matrix with `output_dim` columns.
#' @seealso [simulate_sources()]
#' @examples
#' z <- matrix(rnorm(300), ncol = 3)
#' x <- mix_data(z, n_layers = 3)
#' x5 <- mix_data(z, n_layers = 2, output_dim = 5)
#'
#' # Poisson counts
#' rates <- mix_data(z, n_layers = 2, final_activation = "softplus")
#' counts <- matrix(rpois(length(rates), rates), nrow = nrow(rates))
#' @export
mix_data <- function(x, n_layers = 1,
                     nonlinearity = c("elu", "lrelu", "xtanh", "smooth_lrelu",
                                      "exp_arctan", "leaky_softplus",
                                      "logsumexp"),
                     output_dim = ncol(x),
                     final_activation = c("none", "exp", "softplus"),
                     normalize = TRUE) {
  nonlinearity <- match.arg(nonlinearity)
  final_activation <- match.arg(final_activation)
  x <- as.matrix(x)
  if (output_dim < ncol(x)) {
    stop("`output_dim` must be at least ncol(x) for an injective mixing.",
         call. = FALSE)
  }
  omega <- switch(nonlinearity,
    elu = function(a) ifelse(a < 0, exp(a) - 1, a),
    lrelu = function(a) ifelse(a < 0, 0.2 * a, a),
    xtanh = function(a) tanh(a) + 0.1 * a,
    smooth_lrelu = function(a) 0.5 * (1.05 * a + 0.95 * sqrt(a^2 + 0.1)),
    exp_arctan = function(a) 0.5 * a * exp(atan(a)),
    leaky_softplus = function(a) 0.1 * a + 0.9 * log1p(exp(a)),
    logsumexp = function(a) log(exp(0.1 * a) + exp(a)) - log(2)
  )
  for (l in seq_len(n_layers)) {
    B <- svd(matrix(runif(output_dim * ncol(x), -1, 1), output_dim))$u
    x <- x %*% t(B)
    if (normalize) x <- x / mean(apply(x, 2, sd))
    if (l < n_layers) x <- omega(x)
  }
  x <- switch(final_activation,
    none = x,
    exp = exp(x),
    softplus = log1p(exp(x))
  )
  colnames(x) <- paste0("X", seq_len(ncol(x)))
  x
}

#' Absolute Mean Correlation Coefficient
#'
#' Computes the mean absolute correlation coefficient (MCC) between estimated
#' and true components, a common performance measure in nonlinear BSS.
#'
#' The MCC is
#' \deqn{MCC(K) = \frac{1}{p} \max_{P} \mathrm{tr}(P |K|),}{
#' MCC(K) = (1/p) max_P tr(P |K|),}
#' where \eqn{K} is the \eqn{p \times p} correlation matrix, \eqn{|K|} its
#' elementwise absolute value and the maximum is over permutation matrices
#' \eqn{P}. The maximisation is a linear assignment problem, solved with the
#' Hungarian algorithm. For a non-square matrix, \eqn{p} is the smaller
#' dimension.
#'
#' @param x A correlation matrix, or a matrix of estimated components.
#' @param y Optional matrix of true components. If given, the correlation
#'   matrix `cor(x, y)` is used.
#' @return The MCC, a number between 0 and 1.
#' @examples
#' z <- matrix(rnorm(300), ncol = 3)
#' z_hat <- z[, c(2, 3, 1)] + matrix(rnorm(300, sd = 0.3), ncol = 3)
#' absolute_mean_correlation(z_hat, z)
#' absolute_mean_correlation(cor(z_hat, z))
#' @export
absolute_mean_correlation <- function(x, y = NULL) {
  K <- if (is.null(y)) as.matrix(x) else cor(as.matrix(x), as.matrix(y))
  if (nrow(K) > ncol(K)) K <- t(K)
  -RcppHungarian::HungarianSolver(-abs(K))$cost / min(dim(K))
}
