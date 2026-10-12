#' Identifiable Variational Autoencoder with Autoregressive Latent Components
#'
#' Fits an identifiable variational autoencoder whose latent components follow
#' nonstationary autoregressive processes (iVAEar). The method is meant for
#' time series, spatio-temporal data and other data with a time component,
#' such as panel data \insertCite{sipila2025ivaear}{NonlinearBSS}.
#'
#' @details
#' iVAEar extends [iVAE()] by an autoregressive prior. For an observation of
#' series \eqn{s} (a location or a group) at time \eqn{t}, the auxiliary
#' function gives the trend \eqn{\mu_i(s, t)}, the variance
#' \eqn{\sigma^2_i(s, t)} and the autoregressive coefficients
#' \eqn{\gamma_{i,r}(s, t)}, \eqn{r = 1, \dots, R}, of each latent
#' component, and the prior is
#' \deqn{z_i(s, t) = \mu_i(s, t) + \sum_{r=1}^R \gamma_{i,r}(s, t)
#' (z_i(s, t - r) - \mu_i(s, t - r)) + \epsilon_i(s, t),}{
#' z_i(s,t) = mu_i(s,t) + sum_r gamma_ir(s,t) (z_i(s,t-r) - mu_i(s,t-r)) + eps_i(s,t),}
#' where \eqn{\epsilon_i(s, t)} has variance \eqn{\sigma^2_i(s, t)}. The
#' previous values \eqn{z_i(s, t - r)} are given by the encoder applied to
#' the previous observations of the same series. The components are
#' identifiable when either the variances or the autoregressive
#' coefficients vary sufficiently \insertCite{sipila2025ivaear}{NonlinearBSS}.
#' Temporal dependence as a source of identifiability in nonlinear ICA goes
#' back to \insertCite{HyvarinenMorioka2017;textual}{NonlinearBSS}. The
#' spatio-temporal auxiliary data are the same as for the spatio-temporal
#' iVAE in the Neural Networks article of
#' \insertCite{sipila2024modelling;textual}{NonlinearBSS}.
#'
#' The lag \eqn{r} refers to the \eqn{r}-th previous distinct time point in
#' the data. The series are identified by `group` if given, otherwise by the
#' spatial coordinates; without either the data are a single time series.
#' Observations without the needed previous observations (the first time
#' points, or after gaps) use the trend only. The rows of `x` can be in any
#' order.
#'
#' The fitted trend, variance and autoregressive coefficient functions can
#' be evaluated at new locations and time points, and the latent components
#' can be forecast, using [predict()][predict.nlbss].
#'
#' See [iVAE()] for the supported data formats, auxiliary data and missing
#' values.
#'
#' @inheritParams iVAE
#' @param ar_order The autoregressive order \eqn{R}.
#' @return An object of class `c("iVAEar", "iVAE", "nlbss")`, see [iVAE()].
#'   In addition, the element `lags` gives for each observation the rows of
#'   its previous observations.
#' @seealso [iVAE()], [predict.nlbss()], [aux_rbf()], [aux_segments()]
#' @references \insertAllCited{}
#' @examplesIf tf_available()
#' # Spatio-temporal data: 50 locations observed at 100 time points
#' set.seed(1)
#' n_s <- 50
#' n_t <- 100
#' coords <- matrix(runif(2 * n_s), ncol = 2)[rep(1:n_s, n_t), ]
#' time <- rep(1:n_t, each = n_s)
#' src <- simulate_sources(3, coords = coords, time = time,
#'                         n_segments_space = 5, n_segments_time = 10,
#'                         ar_range = c(-0.8, 0.8))
#' x <- mix_data(src$sources, n_layers = 2)
#'
#' fit <- iVAEar(x, aux_rbf(spatial_basis = c(2, 5), temporal_basis = c(5, 10)),
#'               coords = coords, time = time, epochs = 5, verbose = 0, seed = 1)
#' summary(fit)
#' absolute_mean_correlation(fit$S, src$sources)
#'
#' # Forecast the components for the next two time points
#' new_coords <- coords[1:(2 * n_s), ]
#' fc <- predict(fit, type = "forecast", coords = new_coords,
#'               time = rep(n_t + 1:2, each = n_s))
#' dim(fc)
#' @export
iVAEar <- function(x, aux = NULL, latent_dim = NULL, ar_order = 1,
                   coords = NULL, time = NULL, group = NULL, vars = NULL,
                   batch_aux = FALSE, hidden_units = c(128, 128, 128),
                   aux_hidden_units = c(128, 128, 128),
                   activation = "leaky_relu",
                   source_dist = c("gaussian", "laplace"),
                   error_dist = c("gaussian", "laplace"),
                   error_dist_sigma = 0.01, mask_encoder = TRUE,
                   optimizer = NULL, lr_start = 0.001, lr_end = 0.0001,
                   steps = 10000, epochs = 100, batch_size = 64,
                   validation_split = 0, validation_data = NULL,
                   callbacks = NULL, seed = NULL, verbose = 1) {
  check_common_args(epochs, batch_size, validation_split)
  check_validation(validation_data, validation_split)
  if (!is.numeric(ar_order) || length(ar_order) != 1 || ar_order < 1) {
    stop("`ar_order` must be a positive integer.", call. = FALSE)
  }
  dat <- nlbss_data(x, coords = coords, time = time, group = group,
                    vars = vars, aux_cols = aux_columns(aux))
  fit_vae_family(
    "iVAEar", dat, aux, batch_aux, latent_dim, ar_order = as.integer(ar_order),
    hidden_units = hidden_units, aux_hidden_units = aux_hidden_units,
    activation = activation, source_dist = match.arg(source_dist),
    error_dist = match.arg(error_dist), error_dist_sigma = error_dist_sigma,
    mask_encoder = mask_encoder, optimizer = optimizer, lr_start = lr_start,
    lr_end = lr_end, steps = steps, epochs = epochs, batch_size = batch_size,
    validation_split = validation_split, seed = seed, verbose = verbose,
    call = match.call(), data_name = deparse1(substitute(x)),
    vdat = validation_nlbss(validation_data, dat, coords, time, group,
                            aux_columns(aux)), callbacks = callbacks
  )
}
