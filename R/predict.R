#' Predictions from Nonlinear BSS Models
#'
#' Computes latent components for new data, reconstructions, and the trend,
#' variance and autoregressive coefficient functions of iVAE and iVAEar
#' models at new locations and time points, and forecasts the latent
#' components of iVAEar models.
#'
#' @details
#' The available types are
#' \describe{
#'   \item{`"components"`}{Latent components of the observations in `newdata`
#'   (encoder means). Available for all methods.}
#'   \item{`"reconstruction"`}{The observations in `newdata` encoded and
#'   decoded back, on the scale of the data. Not for [TCL()].}
#'   \item{`"trend"`}{The prior mean \eqn{\mu(u)} of the latent components
#'   given by the auxiliary function. For [iVAEar()] this does not include
#'   the autoregressive part. For [GiVAE()] it is the trend \eqn{\tau(u)}
#'   only (models fitted with `trend = TRUE`), without the graph part.}
#'   \item{`"variance"`}{The prior variance of the latent components. iVAE
#'   type models only.}
#'   \item{`"ar_coef"`}{The autoregressive coefficients, one column per
#'   component and lag. [iVAEar()] only.}
#'   \item{`"forecast"`}{The conditional mean of the latent components at new
#'   locations and time points given the training data, computed by
#'   iterating the autoregressive prior. With `se = TRUE` also the forecast
#'   variances. [iVAEar()] only.}
#' }
#' For types other than `"components"` and `"reconstruction"`, `newdata` only needs
#' the coordinates and time points, which can also be given with `coords`
#' and `time` alone. If `newdata`, `coords` and `time` are all missing, the
#' values for the training data are returned.
#'
#' With `space = "data"`, trends and forecasts are mapped to the observed
#' variables with the decoder, which gives interpolations and forecasts of
#' the data. The decoder is applied to the predicted components, so the
#' result is a plug-in prediction rather than the conditional mean of the
#' data.
#'
#' Forecasts use the previous values of the same series (location or group)
#' from the training data and from earlier forecast time points. Series are
#' matched by `group` or, if not given, by exactly equal coordinates.
#'
#' The components are on the standardised scale of the `S` element of the
#' fitted object unless `unscaled = TRUE`.
#'
#' @param object A fitted model from [iVAE()], [iVAEar()], [VAE()] or [TCL()].
#' @param newdata New data in any format accepted by the fitting functions.
#' @param type The type of prediction, see Details.
#' @param coords,time,group Coordinates, time points and series identifiers
#'   of the new data, if not part of `newdata`. See [iVAE()].
#' @param aux For models fitted with auxiliary variables given as matrices or
#'   vectors (rather than column names), their values for the new data: a
#'   matrix, or a list with one element per such specification.
#' @param unscaled Logical. Return components on the scale of the encoder
#'   output instead of the standardised scale?
#' @param space For `type = "trend"` and `"forecast"`: `"latent"` returns the
#'   latent components, `"data"` the corresponding observed variables
#'   obtained with the decoder.
#' @param se Logical. For `type = "forecast"` with `space = "latent"`, return
#'   also the forecast variances?
#' @param format `"input"` returns the result in the format of `newdata` (or
#'   of the training data), `"matrix"` as a matrix.
#' @param ... Not used.
#' @return The predictions in the format given by `format`. For
#'   `type = "forecast"` with `se = TRUE` a list with elements `fit` and
#'   `var`.
#' @seealso [iVAE()], [iVAEar()], [components()][components.nlbss], [decode()]
#' @examplesIf tf_available()
#' set.seed(1)
#' coords <- matrix(runif(2000), ncol = 2)
#' src <- simulate_sources(3, coords = coords, n_segments_space = 10)
#' x <- mix_data(src$sources, n_layers = 2)
#' fit <- iVAE(x, aux_rbf(c(2, 9)), coords = coords, epochs = 5, verbose = 0)
#'
#' # components of new observations
#' head(predict(fit, x[1:5, ], coords = coords[1:5, ]))
#' # trend and variance of the components on a grid
#' grid <- as.matrix(expand.grid(seq(0, 1, 0.1), seq(0, 1, 0.1)))
#' trend <- predict(fit, type = "trend", coords = grid)
#' # interpolation of the observed variables on the grid
#' x_grid <- predict(fit, type = "trend", coords = grid, space = "data")
#' @export
predict.nlbss <- function(object, newdata = NULL,
                          type = c("components", "reconstruction", "trend", "variance",
                                   "ar_coef", "forecast"),
                          coords = NULL, time = NULL, group = NULL, aux = NULL,
                          space = c("latent", "data"), unscaled = FALSE,
                          se = FALSE, format = c("input", "matrix"), ...) {
  type <- match.arg(type)
  format <- match.arg(format)
  space <- match.arg(space)
  check_type(object, type)
  if (space == "data") {
    if (!type %in% c("trend", "forecast")) {
      stop("`space = \"data\"` is available for `type = \"trend\"` and ",
           "`\"forecast\"`.", call. = FALSE)
    }
    if (se) {
      stop("Forecast variances are only available for `space = \"latent\"`.",
           call. = FALSE)
    }
  }
  no_new <- is.null(newdata) && is.null(coords) && is.null(time)

  if (type %in% c("components", "reconstruction")) {
    if (no_new) {
      return(if (type == "components") components(object, format, unscaled) else fitted(object, format))
    }
    ndat <- nlbss_data(newdata, coords = coords, time = time, group = group,
                       vars = prediction_vars(object, newdata),
                       aux_cols = aux_columns(object$aux$specs))
    if (ncol(ndat$X) != object$p) {
      stop("`newdata` must have ", object$p, " variables.", call. = FALSE)
    }
    Z <- encode_data(object, ndat, aux)
    if (type == "components") {
      M <- if (unscaled) Z else standardize(Z, object$S_center, object$S_scale)
      colnames(M) <- paste0("IC", seq_len(ncol(M)))
    } else {
      M <- decode_unscaled(object, Z)
    }
    return(restore_format(ndat, M, format))
  }

  if (no_new && type != "forecast") {
    ndat <- object$data
    pr <- object$prior
  } else {
    if (no_new) stop("Give the new locations or time points to forecast.", call. = FALSE)
    if (is.null(newdata)) newdata <- matrix(0, nrow = new_rows(coords, time), ncol = 0)
    ndat <- nlbss_data(newdata, coords = coords, time = time, group = group,
                       vars = character(0),
                       aux_cols = aux_columns(object$aux$specs),
                       require_x = FALSE)
    A <- aux_inputs(object$aux, ndat, values = aux, training = FALSE)$raw
    pr <- run_prior(get_net(object), A)
  }
  L <- object$latent_dim
  ic_names <- paste0("IC", seq_len(L))
  M <- switch(type,
    trend = {
      if (space == "data") {
        return(restore_format(ndat, decode_unscaled(object, pr$mean), format))
      }
      out <- if (unscaled) pr$mean else standardize(pr$mean, object$S_center, object$S_scale)
      `colnames<-`(out, ic_names)
    },
    variance = {
      out <- prior_variance(object, pr$log_var)
      if (!unscaled) out <- sweep(out, 2, object$S_scale^2, "/")
      `colnames<-`(out, ic_names)
    },
    ar_coef = {
      out <- do.call(cbind, pr$ar)
      `colnames<-`(out, paste0(rep(ic_names, object$ar_order), "_lag",
                               rep(seq_len(object$ar_order), each = L)))
    },
    forecast = {
      fc <- forecast_ar(object, ndat, pr)
      if (space == "data") {
        return(restore_format(ndat, decode_unscaled(object, fc$fit), format))
      }
      if (!unscaled) {
        fc$fit <- standardize(fc$fit, object$S_center, object$S_scale)
        fc$var <- sweep(fc$var, 2, object$S_scale^2, "/")
      }
      colnames(fc$fit) <- colnames(fc$var) <- ic_names
      if (se) {
        return(list(fit = restore_format(ndat, fc$fit, format),
                    var = restore_format(ndat, fc$var, format)))
      }
      fc$fit
    }
  )
  restore_format(ndat, M, format)
}

check_type <- function(object, type) {
  allowed <- switch(object$method,
    TCL = "components",
    VAE = c("components", "reconstruction"),
    iVAE = c("components", "reconstruction", "trend", "variance"),
    iVAEar = c("components", "reconstruction", "trend", "variance", "ar_coef", "forecast"),
    GiVAE = c("components", "reconstruction", "variance",
             if (isTRUE(object$spec$trend)) "trend")
  )
  if (!type %in% allowed) {
    stop("`type = \"", type, "\"` is not available for ", object$method,
         " models.", call. = FALSE)
  }
}

new_rows <- function(coords, time) {
  if (!is.null(coords)) NROW(coords) else length(time)
}

# Variables of new data: by name for data frames, otherwise all columns.
prediction_vars <- function(object, newdata) {
  if (is.data.frame(newdata) && all(object$var_names %in% names(newdata))) {
    return(object$var_names)
  }
  NULL
}

prior_variance <- function(object, log_var) {
  if (object$spec$source_dist == "laplace") 2 * exp(log_var)^2 else exp(log_var)
}

encode_data <- function(object, ndat, aux) {
  net <- get_net(object)
  Xs <- standardize(ndat$X, object$data_center, object$data_scale)
  mask <- (!is.na(Xs)) * 1
  Xs[mask == 0] <- 0
  if (object$method == "TCL") {
    if (any(mask == 0)) stop("TCL does not allow missing values.", call. = FALSE)
    return(predict_np(net$feature, Xs))
  }
  A <- if (object$method != "VAE") {
    aux_inputs(object$aux, ndat, values = aux, training = FALSE)$raw
  }
  run_encoder(net, object$spec, Xs, A, mask)$mean
}

decode_unscaled <- function(object, Z) {
  X <- run_decoder(get_net(object), Z)
  if (object$spec$error_dist == "poisson") X <- exp(X)
  X <- unstandardize(X, object$data_center, object$data_scale)
  colnames(X) <- object$var_names
  X
}

# Forecasting with the autoregressive prior. The variance recursion keeps,
# for every forecast observation i, Cov(z_i, z_lag_k(i)) for k = 0, ..., R - 1,
# and is exact under the fitted Gaussian AR model with known training values.
forecast_ar <- function(object, ndat, pr) {
  R <- object$ar_order
  L <- object$latent_dim
  n <- object$n
  m <- nrow(ndat$X)
  key <- c(series_key(object$data), series_key(ndat))
  if (is.null(ndat$time)) stop("Forecasting needs time points.", call. = FALSE)
  lags <- lag_index(c(object$data$time, ndat$time), key, R)

  mu <- rbind(object$prior$mean, pr$mean)
  z <- rbind(object$S_unscaled, matrix(NA_real_, m, L))
  s2 <- prior_variance(object, pr$log_var)
  cov <- replicate(R, matrix(0, n + m, L), simplify = FALSE)
  rows <- function(M, idx) {
    out <- matrix(0, length(idx), L)
    ok <- !is.na(idx)
    out[ok, ] <- M[idx[ok], , drop = FALSE]
    out
  }
  # Cov(z_a, z_b) where b is the k-th lag of a
  cov_lag <- function(a, k) rows(cov[[k + 1]], a)

  rank <- match(ndat$time, sort(unique(ndat$time)))
  for (tr in sort(unique(rank))) {
    i <- n + which(rank == tr)
    l <- lags[i, , drop = FALSE]
    g <- lapply(seq_len(R), function(r) pr$ar[[r]][i - n, , drop = FALSE])
    pred <- mu[i, , drop = FALSE]
    for (r in seq_len(R)) {
      ok <- !is.na(l[, r])
      pred[ok, ] <- pred[ok, ] + g[[r]][ok, , drop = FALSE] *
        (z[l[ok, r], , drop = FALSE] - mu[l[ok, r], , drop = FALSE])
    }
    if (anyNA(pred)) stop("Forecasts depend on missing values.", call. = FALSE)
    z[i, ] <- pred

    # covariance of the lagged values: C[r, q] = Cov(z_l_r, z_l_q)
    C <- function(r, q) {
      if (r == q) return(cov_lag(l[, r], 0))
      if (r < q) cov_lag(l[, r], q - r) else cov_lag(l[, q], r - q)
    }
    v <- s2[i - n, , drop = FALSE]
    for (r in seq_len(R)) for (q in seq_len(R)) v <- v + g[[r]] * g[[q]] * C(r, q)
    cov[[1]][i, ] <- v
    for (k in seq_len(R - 1)) {
      ck <- matrix(0, length(i), L)
      for (r in seq_len(R)) ck <- ck + g[[r]] * C(r, k)
      cov[[k + 1]][i, ] <- ck
    }
  }
  list(fit = z[n + seq_len(m), , drop = FALSE],
       var = cov[[1]][n + seq_len(m), , drop = FALSE])
}

#' Map Latent Components to the Data Space
#'
#' Applies the trained decoder, the estimate of the mixing function, to
#' given latent components.
#'
#' @param object A fitted model from [iVAE()], [iVAEar()] or [VAE()].
#' @param S A matrix of latent components, one column per component.
#' @param unscaled Logical. Are the components on the unscaled encoder scale
#'   (`TRUE`) or on the standardised scale of the `S` element of the fitted
#'   object (`FALSE`)?
#' @return A matrix of observations on the scale of the original data.
#' @seealso [predict.nlbss()]
#' @examplesIf tf_available()
#' x <- mix_data(matrix(rnorm(3000), ncol = 3), n_layers = 2)
#' fit <- VAE(x, epochs = 2, verbose = 0)
#' decode(fit, matrix(0, 1, 3))
#' @export
decode <- function(object, S, unscaled = FALSE) {
  if (!inherits(object, "nlbss") || object$method == "TCL") {
    stop("`object` must be a model fitted by iVAE(), iVAEar() or VAE().",
         call. = FALSE)
  }
  S <- as.matrix(S)
  if (ncol(S) != object$latent_dim) {
    stop("`S` must have ", object$latent_dim, " columns.", call. = FALSE)
  }
  Z <- if (unscaled) S else unstandardize(S, object$S_center, object$S_scale)
  decode_unscaled(object, Z)
}