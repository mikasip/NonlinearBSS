#' Identifiable Variational Autoencoder
#'
#' Fits an identifiable variational autoencoder (iVAE)
#' \insertCite{Khemakhem2020}{NonlinearBSS} for nonlinear blind source
#' separation. The method can be used for any data for which auxiliary
#' variables explaining the nonstationarity of the latent components are
#' available, in particular for time series, spatial
#' \insertCite{sipila2024spatial}{NonlinearBSS} and spatio-temporal data
#' \insertCite{sipila2024modelling}{NonlinearBSS}.
#'
#' @details
#' The observed data \eqn{x} are assumed to be generated as \eqn{x = f(z)},
#' where \eqn{f} is an unknown injective mixing function and the latent
#' components \eqn{z_1, \dots, z_L} are independent given auxiliary variables
#' \eqn{u}. iVAE consists of an encoder \eqn{g(x, u)} giving the mean and
#' variance of the approximate posterior \eqn{q(z | x, u)}, a decoder
#' \eqn{h(z)} estimating the inverse of the encoder (the mixing function), and
#' an auxiliary function \eqn{w(u)} giving the mean and variance of the prior
#' \eqn{p(z | u)}. All three are fully connected neural networks, trained by
#' maximising the evidence lower bound
#' \deqn{E_{q(z|x,u)}[\log p(x | z) + \log p(z | u) - \log q(z | x, u)].}{
#' E_q(z|x,u) [log p(x|z) + log p(z|u) - log q(z|x,u)].}
#' The first term is the reconstruction accuracy, controlled by `error_dist`
#' and `error_dist_sigma`: decreasing `error_dist_sigma` gives more weight to
#' the reconstruction. The remaining terms make the posterior close to the
#' prior, which leads to independent components. When the variances of the
#' latent components vary sufficiently with \eqn{u}, the components are
#' identifiable up to permutation and componentwise transformations
#' \insertCite{Khemakhem2020,Hyvarinen2019}{NonlinearBSS}. For spatial data,
#' segmentations and radial basis functions of the coordinates are suitable
#' auxiliary variables \insertCite{sipila2024spatial}{NonlinearBSS}, and for
#' spatio-temporal data the corresponding spatio-temporal versions
#' \insertCite{sipila2024modelling}{NonlinearBSS}.
#'
#' The estimated components are the encoder means for the observed data,
#' standardised to zero mean and unit variance.
#'
#' **Data formats.** `x` can be a numeric matrix or data frame, a time series
#' (`ts`, `xts`, `zoo`, `tsibble`) or a spatial or spatio-temporal object
#' (`sf`, `sftime`, `STFDF`). Coordinates and time points are taken from the
#' object; for matrices and data frames they can be given with `coords` and
#' `time`. The results can be extracted in the input format with [components()][components.nlbss].
#'
#' **Auxiliary data.** `aux` is a matrix of auxiliary variables or a
#' specification created by [aux_segments()], [aux_rbf()], [aux_coords()] or
#' [aux_covariates()], or a list combining several of these. If `aux` is
#' not given and the data have coordinates or time points, radial basis
#' functions [aux_rbf()] are used. With `batch_aux = TRUE` all auxiliary
#' variables (radial basis functions, segment indicators, coordinates and
#' covariates) are computed batch-wise inside the network from the raw
#' coordinates, time points and covariates, instead of storing the full
#' auxiliary matrix. This gives the same auxiliary variables but saves memory
#' for large data.
#'
#' **Validation and early stopping.** The training can be monitored on a
#' validation set given by `validation_data`, for example a later time
#' period. For [iVAEar()] the first validation time points use the end of the
#' training data as previous values. Keras callbacks, such as
#' `keras3::callback_early_stopping()`, are passed with `callbacks`.
#'
#' **Missing values** in `x` are allowed. The reconstruction loss ignores
#' them and, if `mask_encoder = TRUE`, the missingness indicators are given to
#' the encoder as extra inputs. Together with covariates as auxiliary
#' variables ([aux_covariates()]), for example satellite-derived predictors,
#' this allows modelling incomplete monitoring data
#' \insertCite{sipila2026missing}{NonlinearBSS}.
#'
#' @param x The observed data. See Details for the supported formats.
#' @param aux Auxiliary data: a numeric matrix with one row per observation,
#'   a specification such as [aux_rbf()] or [aux_segments()], or a list of
#'   these. See Details.
#' @param latent_dim The number of latent components. Defaults to the number
#'   of observed variables.
#' @param coords Spatial coordinates, a matrix with one row per observation
#'   or names of columns in `x`. Taken from the object for spatial formats.
#' @param time Time points (numeric, `Date` or `POSIXct`) or the name of a
#'   column in `x`. Taken from the object for time series formats.
#' @param group Identifier of the series an observation belongs to (for
#'   example a station), or names of columns in `x`. Only needed for
#'   [iVAEar()] when the series are not identified by the coordinates.
#' @param vars Names or indices of the observed variables. Defaults to all
#'   numeric columns not used for `coords`, `time`, `group` or `aux`.
#' @param batch_aux Logical. Compute the auxiliary variables batch-wise
#'   inside the network?
#' @param hidden_units Numbers of units in the hidden layers of the encoder
#'   (the decoder uses the reverse order).
#' @param aux_hidden_units Numbers of units in the hidden layers of the
#'   auxiliary function.
#' @param activation Activation function of the hidden layers.
#' @param source_dist Distribution of the latent components given the
#'   auxiliary variables, `"gaussian"` or `"laplace"`.
#' @param error_dist Distribution of the reconstruction error: `"gaussian"` or
#'   `"laplace"`.
#' @param error_dist_sigma Scale of the reconstruction error (the variance for
#'   `"gaussian"`). A vector of length two defines a linear schedule from the
#'   first to the second value over `steps` training steps.
#' @param mask_encoder Logical. Give the missing value indicators to the
#'   encoder when `x` contains missing values?
#' @param optimizer A Keras optimizer. Defaults to Adam with a polynomial
#'   learning rate decay from `lr_start` to `lr_end` over `steps` steps.
#' @param lr_start,lr_end Initial and final learning rate of the default
#'   optimizer.
#' @param steps Number of training steps (batches) for the learning rate
#'   decay and the `error_dist_sigma` schedule.
#' @param epochs Number of training epochs.
#' @param batch_size Batch size.
#' @param validation_split Proportion of the observations used for
#'   validation. Keras takes the last observations, not a random subset.
#' @param validation_data Optional validation data, used instead of
#'   `validation_split`: either data in the same format as `x` (column names
#'   given in `coords`, `time` and `group` are used for it as well), or a list
#'   with elements `x` and, as needed, `coords`, `time`, `group` and `aux`
#'   (values of auxiliary variables given as matrices, see [predict.nlbss()]).
#' @param callbacks A list of Keras callbacks, for example
#'   `list(keras3::callback_early_stopping(patience = 5))`.
#' @param seed Optional random seed for reproducible training. Use this
#'   instead of [set.seed()]; the random number state of the R session is
#'   restored after fitting.
#' @param verbose Verbosity of the training: 0 (silent), 1 (progress bar) or
#'   2 (one line per epoch).
#'
#' @return An object of class `c("iVAE", "nlbss")`, a list with elements
#'   \item{S}{Matrix of the estimated latent components, standardised.}
#'   \item{S_unscaled}{The encoder means before standardisation.}
#'   \item{S_center, S_scale}{Means and standard deviations used in the
#'   standardisation.}
#'   \item{elbo}{The evidence lower bound of the fitted model, evaluated at
#'   the encoder means.}
#'   \item{history}{Training metrics for each epoch.}
#'   \item{aux}{The fitted auxiliary data specifications.}
#'   \item{prior}{Prior means, log variances and AR coefficients for the
#'   training data (unscaled).}
#'   \item{data_center, data_scale}{Means and standard deviations used to
#'   standardise the data.}
#'   \item{data}{Coordinates, time points and format of the data.}
#'   \item{spec, weights}{Network specification and trained weights.}
#'   \item{call, data_name}{The function call and the name of the data.}
#'
#' Use [components()][components.nlbss] to get the components in the input format, and [predict()][predict.nlbss],
#' [fitted()][fitted.nlbss], [summary()][summary.nlbss] and [plot()][plot.nlbss] for further analysis.
#'
#' @seealso [iVAEar()], [VAE()], [TCL()], [aux_rbf()], [aux_segments()],
#'   [predict.nlbss()], [save_nlbss()]
#' @references \insertAllCited{}
#' @examplesIf tf_available()
#' # Latent components with variance changing over segments
#' set.seed(1)
#' src <- simulate_sources(3, time = 1:1000, n_segments_time = 10)
#' x <- mix_data(src$sources, n_layers = 2)
#'
#' # Auxiliary data: segment indicators of the time series
#' fit <- iVAE(x, aux = aux_segments(100), time = 1:1000, epochs = 10,
#'             verbose = 0, seed = 1)
#' fit
#' absolute_mean_correlation(fit$S, src$sources)
#'
#' # Spatial data with radial basis functions as auxiliary data
#' coords <- matrix(runif(2000), ncol = 2)
#' src <- simulate_sources(3, coords = coords, n_segments_space = 10)
#' x <- mix_data(src$sources, n_layers = 2)
#' fit <- iVAE(x, aux_rbf(spatial_basis = c(2, 9)), coords = coords,
#'             epochs = 10, verbose = 0, seed = 1)
#' absolute_mean_correlation(fit$S, src$sources)
#' @export
iVAE <- function(x, aux = NULL, latent_dim = NULL, coords = NULL, time = NULL,
                 group = NULL, vars = NULL, batch_aux = FALSE,
                 hidden_units = c(128, 128, 128),
                 aux_hidden_units = c(128, 128, 128),
                 activation = "leaky_relu",
                 source_dist = c("gaussian", "laplace"),
                 error_dist = c("gaussian", "laplace"),
                 error_dist_sigma = 0.02, mask_encoder = TRUE,
                 optimizer = NULL, lr_start = 0.001, lr_end = 0.0001,
                 steps = 10000, epochs = 100, batch_size = 64,
                 validation_split = 0, validation_data = NULL,
                 callbacks = NULL, seed = NULL, verbose = 1) {
  check_common_args(epochs, batch_size, validation_split)
  check_validation(validation_data, validation_split)
  dat <- nlbss_data(x, coords = coords, time = time, group = group,
                    vars = vars, aux_cols = aux_columns(aux))
  fit_vae_family(
    "iVAE", dat, aux, batch_aux, latent_dim, ar_order = 0,
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
