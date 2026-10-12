# Common fitting routine of VAE, iVAE and iVAEar.

fit_vae_family <- function(method, dat, aux, batch_aux, latent_dim, ar_order,
                           hidden_units, aux_hidden_units, activation,
                           source_dist, error_dist, error_dist_sigma,
                           mask_encoder, optimizer, lr_start, lr_end, steps,
                           epochs, batch_size, validation_split, seed, verbose,
                           call, data_name, vdat = NULL, callbacks = NULL) {
  check_tf()
  if (!is.null(seed)) {
    old_rng <- set_seed(seed)
    on.exit(restore_rng(old_rng), add = TRUE)
  }
  check_sigma(error_dist_sigma)
  n <- nrow(dat$X)
  p <- ncol(dat$X)
  latent_dim <- if (is.null(latent_dim)) p else as.integer(latent_dim)
  if (method != "VAE" && is.null(aux)) aux <- default_aux(dat)

  sc <- scale_data(dat$X, error_dist)
  has_missing <- any(sc$mask == 0)

  prep <- NULL
  A <- NULL
  if (method != "VAE") {
    prep <- prepare_aux(aux, dat, batch = batch_aux)
    A <- prep$raw
    if (nrow(A) != n) stop("The auxiliary data must have one row per observation.", call. = FALSE)
    if (anyNA(A)) stop("The auxiliary data must not contain missing values.", call. = FALSE)
    prep$raw <- NULL
  }

  lags <- NULL
  if (ar_order > 0) {
    if (is.null(dat$time)) {
      stop("iVAEar needs time points: give `time` or use a time series or ",
           "spatio-temporal data format.", call. = FALSE)
    }
    lags <- lag_index(dat$time, series_key(dat), ar_order)
  }

  spec <- list(
    method = method, p = p, latent_dim = latent_dim,
    n_aux = if (is.null(A)) 0L else ncol(A),
    layer_config = prep$layer_config,
    hidden_units = as.integer(hidden_units),
    aux_hidden_units = as.integer(aux_hidden_units), activation = activation,
    source_dist = source_dist, error_dist = error_dist,
    error_dist_sigma = error_dist_sigma, steps = steps,
    ar_order = as.integer(ar_order), use_mask = has_missing && mask_encoder,
    seed = seed
  )
  if (is.null(optimizer)) optimizer <- default_optimizer(lr_start, lr_end, steps)
  net <- build_network(spec, optimizer$iterations)
  keras3::compile(net$train, optimizer = optimizer, loss = elbo_loss,
                  metrics = elbo_metrics())

  inputs <- network_inputs(sc$X, sc$mask, A, lags)
  validation <- NULL
  if (!is.null(vdat)) {
    if (ncol(vdat$X) != p) {
      stop("`validation_data` must have ", p, " variables.", call. = FALSE)
    }
    Xv <- standardize(vdat$X, sc$center, sc$scale)
    Mv <- (!is.na(Xv)) * 1
    Xv[Mv == 0] <- 0
    Av <- if (!is.null(A)) {
      aux_inputs(prep, vdat, values = vdat$aux_values, training = FALSE)$raw
    }
    rows <- n + seq_len(nrow(Xv))
    # lags of the validation data may refer to the end of the training data
    vlags <- if (ar_order > 0) {
      lag_index(c(dat$time, vdat$time), c(series_key(dat), series_key(vdat)),
                ar_order)
    }
    validation <- list(
      network_inputs(rbind(sc$X, Xv), rbind(sc$mask, Mv), rbind(A, Av),
                     vlags, rows),
      matrix(0, length(rows), 1)
    )
  }
  history <- keras3::fit(
    net$train, inputs, matrix(0, n, 1), epochs = as.integer(epochs),
    batch_size = as.integer(batch_size), validation_split = validation_split,
    validation_data = validation, callbacks = callbacks,
    shuffle = TRUE, verbose = verbose
  )

  enc <- run_encoder(net, spec, sc$X, A, sc$mask)
  Z <- enc$mean
  prior <- NULL
  if (method != "VAE") {
    prior <- run_prior(net, A)
    prior_mean <- if (ar_order > 0) ar_prior_mean(prior$mean, prior$ar, Z, lags) else prior$mean
    prior_log_var <- prior$log_var
  } else {
    prior_mean <- matrix(0, n, latent_dim)
    prior_log_var <- matrix(0, n, latent_dim)
  }
  X_hat <- run_decoder(net, Z)
  sigma_end <- error_dist_sigma[length(error_dist_sigma)]
  log_px <- rowSums(r_log_density(error_dist, sc$X, X_hat, sigma_end) * sc$mask)
  log_qz <- rowSums(r_log_density(source_dist, Z, Z, exp(enc$log_var)))
  log_pz <- rowSums(r_log_density(source_dist, Z, prior_mean, exp(prior_log_var)))

  S_center <- colMeans(Z)
  S_scale <- safe_range(apply(Z, 2, sd))
  S <- standardize(Z, S_center, S_scale)
  colnames(Z) <- colnames(S) <- paste0("IC", seq_len(latent_dim))

  models <- new.env(parent = emptyenv())
  models$net <- net
  structure(list(
    S = S,
    S_unscaled = Z,
    S_center = S_center,
    S_scale = S_scale,
    method = method,
    latent_dim = latent_dim,
    ar_order = as.integer(ar_order),
    n = n,
    p = p,
    var_names = colnames(dat$X),
    data_center = sc$center,
    data_scale = sc$scale,
    has_missing = has_missing,
    data = strip_data(dat),
    aux = prep,
    lags = lags,
    prior = prior,
    elbo = mean(log_px + log_pz - log_qz),
    history = history_df(history),
    spec = spec,
    weights = get_trainable_weights(net$train),
    models = models,
    call = call,
    data_name = data_name
  ), class = c(if (method == "iVAEar") "iVAEar",
               if (method != "VAE") "iVAE" else "VAE", "nlbss"))
}

check_sigma <- function(sigma) {
  if (!is.numeric(sigma) || !length(sigma) %in% 1:2 || any(sigma <= 0)) {
    stop("`error_dist_sigma` must be a positive number or a vector of two ",
         "positive numbers (start and end of a linear schedule).", call. = FALSE)
  }
}

scale_data <- function(X, error_dist) {
  mask <- (!is.na(X)) * 1
  p <- ncol(X)
  if (error_dist %in% c("poisson", "bernoulli")) {
    if (error_dist == "bernoulli" && !all(X[!is.na(X)] %in% c(0, 1))) {
      stop("`error_dist = \"bernoulli\"` needs binary (0/1) data.", call. = FALSE)
    }
    if (error_dist == "poisson" && any(X < 0, na.rm = TRUE)) {
      stop("`error_dist = \"poisson\"` needs non-negative count data.", call. = FALSE)
    }
    center <- rep(0, p)
    scale <- rep(1, p)
  } else {
    center <- colMeans(X, na.rm = TRUE)
    scale <- safe_range(apply(X, 2, sd, na.rm = TRUE))
  }
  Xs <- standardize(X, center, scale)
  Xs[mask == 0] <- 0
  list(X = Xs, mask = mask, center = center, scale = scale)
}

standardize <- function(X, center, scale) {
  sweep(sweep(X, 2, center, "-"), 2, scale, "/")
}

unstandardize <- function(X, center, scale) {
  sweep(sweep(X, 2, scale, "*"), 2, center, "+")
}

# Keep what is needed for restoring outputs and forecasting, not the data.
strip_data <- function(dat) {
  dat$X <- dat$X[, 0, drop = FALSE]
  dat$template$columns <- NULL
  dat
}

# Make sure the Keras models of a fitted object exist, rebuilding them from
# the stored weights after the object has been saved and loaded.
get_net <- function(object) {
  net <- object$models$net
  if (is.null(net) || reticulate::py_is_null_xptr(net$train)) {
    check_tf()
    spec <- object$spec
    net <- if (inherits(object, "TCL")) {
      build_tcl_network(spec)
    } else {
      build_network(spec, keras3::optimizer_adam()$iterations)
    }
    set_trainable_weights(net$train, object$weights)
    if (is.environment(object$models)) object$models$net <- net
  }
  net
}

check_common_args <- function(epochs, batch_size, validation_split) {
  if (!is.numeric(epochs) || epochs < 1) stop("`epochs` must be a positive integer.", call. = FALSE)
  if (!is.numeric(batch_size) || batch_size < 1) stop("`batch_size` must be a positive integer.", call. = FALSE)
  if (!is.numeric(validation_split) || validation_split < 0 || validation_split >= 1) {
    stop("`validation_split` must be in [0, 1).", call. = FALSE)
  }
}

# Only the trainable weights are stored: the other variables (optimizer step,
# random seed state) are not needed for prediction.
get_trainable_weights <- function(model) {
  lapply(model$trainable_weights, function(v) as.array(v$numpy()))
}

set_trainable_weights <- function(model, weights) {
  vars <- model$trainable_weights
  if (length(vars) != length(weights)) {
    stop("The stored weights do not match the model.", call. = FALSE)
  }
  for (i in seq_along(vars)) vars[[i]]$assign(weights[[i]])
  invisible(model)
}

# Set the random seed (also for Python and TensorFlow if `tf = TRUE`) and
# return the previous state of R's random number generator, so that it can
# be restored: setting a seed must not change the user's random numbers.
set_seed <- function(seed, tf = TRUE) {
  old <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  }
  if (tf) keras3::set_random_seed(as.integer(seed)) else set.seed(seed)
  old
}

restore_rng <- function(old) {
  if (is.null(old)) {
    if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  } else {
    assign(".Random.seed", old, envir = globalenv())
  }
}

# Inputs of the training network for the rows `rows` of the data matrices.
# Lagged inputs of observations without previous values are replaced by the
# observation itself and marked as invalid.
network_inputs <- function(X, mask, A, lags, rows = seq_len(nrow(X))) {
  inputs <- c(list(X[rows, , drop = FALSE], mask[rows, , drop = FALSE]),
              if (!is.null(A)) list(A[rows, , drop = FALSE]))
  for (r in seq_len(if (is.null(lags)) 0 else ncol(lags))) {
    j <- lags[rows, r]
    valid <- !is.na(j)
    j[!valid] <- rows[!valid]
    inputs <- c(inputs, list(X[j, , drop = FALSE], mask[j, , drop = FALSE],
                             A[j, , drop = FALSE], matrix(as.numeric(valid))))
  }
  inputs
}

# Validation data: an object in the same format as the training data (column
# names given in `coords`, `time` and `group` are reused) or a list with
# elements `x` and optionally `coords`, `time`, `group` and `aux`.
validation_nlbss <- function(v, dat, coords, time, group, aux_cols) {
  if (is.null(v)) return(NULL)
  vars <- function(x) {
    if (is.data.frame(x) && all(colnames(dat$X) %in% names(x))) colnames(dat$X)
  }
  if (is.list(v) && !is.data.frame(v) && !is.null(v$x)) {
    vd <- nlbss_data(v$x, coords = v$coords, time = v$time, group = v$group,
                     vars = vars(v$x), aux_cols = aux_cols)
    vd$aux_values <- v$aux
    return(vd)
  }
  names_only <- function(a) if (is.character(a)) a
  nlbss_data(v, coords = names_only(coords), time = names_only(time),
             group = names_only(group), vars = vars(v), aux_cols = aux_cols)
}

check_validation <- function(validation_data, validation_split) {
  if (!is.null(validation_data) && validation_split > 0) {
    stop("Give either `validation_data` or `validation_split`, not both.",
         call. = FALSE)
  }
}

# Training history as a data frame. After early stopping Keras pads the
# history with missing values up to `epochs`; these rows are dropped.
history_df <- function(history) {
  h <- as.data.frame(history)
  h <- h[!is.na(h$value), , drop = FALSE]
  rownames(h) <- NULL
  h
}
