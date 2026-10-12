#' Time Contrastive Learning
#'
#' Estimates latent components of nonstationary time series with time
#' contrastive learning (TCL) \insertCite{HyvarinenMorioka2016}{NonlinearBSS}.
#'
#' @details
#' The time series is divided into segments, and a neural network is trained
#' to classify each observation into the segment it belongs to. The network
#' consists of hidden layers, a linear feature layer with `latent_dim` units,
#' an absolute value nonlinearity and a multinomial logistic regression
#' layer. If the latent components have variances that change between the
#' segments, the feature layer learns the latent components up to
#' componentwise transformations
#' \insertCite{HyvarinenMorioka2016}{NonlinearBSS}.
#'
#' The segments are given by `labels`, which can be a vector of segment
#' labels or an [aux_segments()] specification. If `labels` is not given,
#' the time range is divided into `n_segments` segments of equal length. TCL
#' can also be applied to spatial or spatio-temporal data by giving a
#' suitable segmentation.
#'
#' @inheritParams iVAE
#' @param labels Segment labels, a vector with one value per observation, or
#'   an [aux_segments()] specification.
#' @param n_segments Number of segments of equal length used if `labels` is
#'   not given.
#' @param hidden_units Numbers of units in the hidden layers.
#' @param lr_start,lr_end Initial and final learning rate.
#' @param validation_data Optional validation data, used instead of
#'   `validation_split`: a list with elements `x` (data in any supported
#'   format) and `labels` (its segment labels, using the training labels).
#' @return An object of class `c("TCL", "nlbss")` with elements `S`,
#'   `S_unscaled`, `S_center`, `S_scale`, `labels`, `accuracy` (final
#'   training classification accuracy), `history` and others as in [iVAE()].
#' @seealso [iVAE()], [iVAEar()]
#' @references \insertAllCited{}
#' @examplesIf tf_available()
#' set.seed(1)
#' src <- simulate_sources(3, time = 1:2000, n_segments_time = 20,
#'                         mean_range = c(0, 0))
#' x <- ts(mix_data(src$sources, n_layers = 2))
#' fit <- TCL(x, n_segments = 20, epochs = 10, verbose = 0, seed = 1)
#' fit
#' absolute_mean_correlation(fit$S, src$sources)
#' @export
TCL <- function(x, labels = NULL, latent_dim = NULL, n_segments = 10,
                coords = NULL, time = NULL, group = NULL, vars = NULL,
                hidden_units = 32, activation = "leaky_relu",
                lr_start = 0.01, lr_end = 0.001, steps = 10000, epochs = 100,
                batch_size = 64, validation_split = 0, validation_data = NULL,
                callbacks = NULL, seed = NULL, verbose = 1) {
  check_common_args(epochs, batch_size, validation_split)
  check_validation(validation_data, validation_split)
  dat <- nlbss_data(x, coords = coords, time = time, group = group, vars = vars)
  check_tf()
  if (!is.null(seed)) {
    old_rng <- set_seed(seed)
    on.exit(restore_rng(old_rng), add = TRUE)
  }
  if (anyNA(dat$X)) stop("TCL does not allow missing values.", call. = FALSE)
  n <- nrow(dat$X)
  p <- ncol(dat$X)
  latent_dim <- if (is.null(latent_dim)) p else as.integer(latent_dim)

  if (inherits(labels, "aux_segments")) {
    labels <- segment_labels(fit_aux(labels, dat), dat)
  } else if (is.null(labels)) {
    t <- if (is.null(dat$time)) seq_len(n) else dat$time
    rank <- match(t, sort(unique(t)))
    labels <- cut(rank, breaks = n_segments, labels = FALSE)
  }
  if (length(labels) != n) stop("`labels` must have one value per observation.", call. = FALSE)
  classes <- sort(unique(labels))
  y <- match(labels, classes) - 1L

  center <- colMeans(dat$X)
  scale <- safe_range(apply(dat$X, 2, sd))
  Xs <- standardize(dat$X, center, scale)

  spec <- list(method = "TCL", p = p, latent_dim = latent_dim,
               n_classes = length(classes), hidden_units = as.integer(hidden_units),
               activation = activation)
  net <- build_tcl_network(spec)
  keras3::compile(net$train,
                  optimizer = default_optimizer(lr_start, lr_end, steps),
                  loss = "sparse_categorical_crossentropy",
                  metrics = "accuracy")
  validation <- NULL
  if (!is.null(validation_data)) {
    if (!is.list(validation_data) || is.null(validation_data$x) ||
        is.null(validation_data$labels)) {
      stop("`validation_data` must be a list with elements `x` and `labels`.",
           call. = FALSE)
    }
    vX <- nlbss_data(validation_data$x, vars = vars)$X
    yv <- match(validation_data$labels, classes) - 1L
    if (ncol(vX) != p || length(yv) != nrow(vX) || anyNA(yv) || anyNA(vX)) {
      stop("`validation_data` must have ", p, " variables, no missing ",
           "values and one training label per observation.", call. = FALSE)
    }
    validation <- list(standardize(vX, center, scale), matrix(yv))
  }
  history <- keras3::fit(net$train, Xs, matrix(y), epochs = as.integer(epochs),
                         batch_size = as.integer(batch_size),
                         validation_split = validation_split,
                         validation_data = validation, callbacks = callbacks,
                         shuffle = TRUE, verbose = verbose)
  history <- history_df(history)

  Z <- predict_np(net$feature, Xs)
  S_center <- colMeans(Z)
  S_scale <- safe_range(apply(Z, 2, sd))
  S <- standardize(Z, S_center, S_scale)
  colnames(Z) <- colnames(S) <- paste0("IC", seq_len(latent_dim))
  acc <- history$value[history$metric == "accuracy" & history$data == "training"]

  models <- new.env(parent = emptyenv())
  models$net <- net
  structure(list(
    S = S, S_unscaled = Z, S_center = S_center, S_scale = S_scale,
    method = "TCL", latent_dim = latent_dim, ar_order = 0L, n = n, p = p,
    var_names = colnames(dat$X), data_center = center, data_scale = scale,
    has_missing = FALSE, data = strip_data(dat), labels = labels,
    accuracy = acc[length(acc)], history = history, spec = spec,
    weights = get_trainable_weights(net$train), models = models,
    call = match.call(), data_name = deparse1(substitute(x))
  ), class = c("TCL", "nlbss"))
}

build_tcl_network <- function(spec) {
  x_in <- keras3::layer_input(shape = as.integer(spec$p))
  h <- x_in
  for (units in spec$hidden_units) {
    h <- keras3::layer_dense(h, units = units, activation = spec$activation)
  }
  features <- keras3::layer_dense(h, units = as.integer(spec$latent_dim),
                                  name = "features")
  out <- keras3::layer_lambda(features, f = keras3::op_abs)
  out <- keras3::layer_dense(out, units = as.integer(spec$n_classes),
                             activation = "softmax")
  list(train = keras3::keras_model(x_in, out),
       feature = keras3::keras_model(x_in, features))
}
