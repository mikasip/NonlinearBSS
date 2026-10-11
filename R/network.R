# Construction of the (i)VAE networks.
#
# `spec` is a list with elements
#   method            "VAE", "iVAE" or "iVAEar"
#   p, latent_dim     data and latent dimension
#   n_aux             number of columns of the raw auxiliary input
#   layer_config      configuration for batch-wise auxiliary variables
#   hidden_units, aux_hidden_units, activation
#   source_dist, error_dist, error_dist_sigma, steps
#   ar_order          autoregressive order (0 for VAE and iVAE)
#   use_mask          include the missing value mask in the encoder input
#   seed
#
# The returned list contains the training model and the models used for
# prediction: encoder (mean and log variance), prior (mean, log variance and
# AR coefficients) and decoder.

build_network <- function(spec, step) {
  L <- as.integer(spec$latent_dim)
  p <- as.integer(spec$p)
  R <- as.integer(spec$ar_order)
  has_aux <- spec$method != "VAE"
  act <- spec$activation
  dense <- function(units, activation = NULL) {
    keras3::layer_dense(units = as.integer(units), activation = activation)
  }

  x_in <- keras3::layer_input(shape = p, name = "x")
  mask_in <- keras3::layer_input(shape = p, name = "mask")
  aux_in <- if (has_aux) keras3::layer_input(shape = as.integer(spec$n_aux), name = "aux")

  aux_features <- NULL
  if (!is.null(spec$layer_config)) {
    aux_features <- keras3::layer_lambda(f = aux_tf_function(spec$layer_config),
                                          name = "aux_features")
  }
  features <- function(a) if (is.null(aux_features)) a else aux_features(a)

  # Encoder g(x, u)
  enc_hidden <- lapply(spec$hidden_units, dense, activation = act)
  enc_mean <- dense(L)
  enc_log_var <- dense(L)
  encode <- function(x, f, m) {
    h <- c(list(x), if (has_aux) list(f), if (spec$use_mask) list(m))
    h <- if (length(h) > 1) keras3::layer_concatenate(h) else h[[1]]
    for (layer in enc_hidden) h <- layer(h)
    list(mean = enc_mean(h), log_var = enc_log_var(h))
  }

  # Auxiliary function w(u) giving the prior parameters
  if (has_aux) {
    prior_hidden <- lapply(spec$aux_hidden_units, dense, activation = act)
    prior_mean_layer <- dense(L)
    prior_log_var_layer <- dense(L)
    prior_ar_layers <- lapply(seq_len(R), function(r) dense(L, "tanh"))
    prior <- function(f) {
      h <- f
      for (layer in prior_hidden) h <- layer(h)
      list(mean = prior_mean_layer(h), log_var = prior_log_var_layer(h),
           ar = lapply(prior_ar_layers, function(layer) layer(h)))
    }
  }

  # Decoder h(z)
  dec_hidden <- lapply(rev(spec$hidden_units), dense, activation = act)
  dec_out <- dense(p, if (spec$error_dist == "bernoulli") "sigmoid")
  decode <- function(z) {
    for (layer in dec_hidden) z <- layer(z)
    dec_out(z)
  }

  # Training graph
  f0 <- if (has_aux) features(aux_in)
  enc <- encode(x_in, f0, mask_in)
  sampling <- nlbss_layer("sampling")(dist = spec$source_dist, seed = spec$seed)
  z <- sampling(list(enc$mean, enc$log_var))
  x_hat <- decode(z)

  inputs <- c(list(x_in, mask_in), if (has_aux) list(aux_in))
  if (has_aux) {
    pr <- prior(f0)
    prior_mean <- pr$mean
    for (r in seq_len(R)) {
      x_lag <- keras3::layer_input(shape = p, name = paste0("x_lag", r))
      mask_lag <- keras3::layer_input(shape = p, name = paste0("mask_lag", r))
      aux_lag <- keras3::layer_input(shape = as.integer(spec$n_aux),
                                     name = paste0("aux_lag", r))
      valid_lag <- keras3::layer_input(shape = 1L, name = paste0("valid_lag", r))
      f_lag <- features(aux_lag)
      z_lag <- encode(x_lag, f_lag, mask_lag)$mean
      mean_lag <- prior(f_lag)$mean
      ar_term <- keras3::op_multiply(
        keras3::op_multiply(valid_lag, pr$ar[[r]]),
        keras3::op_subtract(z_lag, mean_lag)
      )
      prior_mean <- keras3::op_add(prior_mean, ar_term)
      inputs <- c(inputs, list(x_lag, mask_lag, aux_lag, valid_lag))
    }
    prior_log_var <- pr$log_var
  } else {
    prior_mean <- keras3::op_zeros_like(enc$mean)
    prior_log_var <- keras3::op_zeros_like(enc$mean)
  }
  sigma <- spec$error_dist_sigma
  elbo <- nlbss_layer("elbo")(
    error_dist = spec$error_dist, source_dist = spec$source_dist,
    sigma_start = sigma[1], sigma_end = sigma[length(sigma)],
    steps = spec$steps, step = step
  )
  out <- elbo(list(x_in, x_hat, mask_in, z, enc$mean, enc$log_var,
                   prior_mean, prior_log_var))
  train <- keras3::keras_model(inputs, out)

  # Prediction models
  enc_inputs <- c(list(x_in), if (has_aux) list(aux_in),
                  if (spec$use_mask) list(mask_in))
  encoder <- keras3::keras_model(enc_inputs, list(enc$mean, enc$log_var))
  z_in <- keras3::layer_input(shape = L)
  decoder <- keras3::keras_model(z_in, decode(z_in))
  prior_model <- if (has_aux) {
    keras3::keras_model(aux_in, c(list(pr$mean, pr$log_var), pr$ar))
  }
  list(train = train, encoder = encoder, decoder = decoder, prior = prior_model)
}

# Inputs of the encoder model in the order used by build_network().
encoder_inputs <- function(spec, X, A, mask) {
  inputs <- c(list(X), if (spec$method != "VAE") list(A), if (spec$use_mask) list(mask))
  if (length(inputs) == 1) inputs[[1]] else inputs
}

# Run the prediction models; returns plain R matrices.
run_encoder <- function(net, spec, X, A, mask) {
  out <- predict_np(net$encoder, encoder_inputs(spec, X, A, mask))
  list(mean = out[[1]], log_var = out[[2]])
}

run_prior <- function(net, A) {
  out <- predict_np(net$prior, A)
  if (!is.list(out)) out <- list(out)
  list(mean = out[[1]], log_var = out[[2]], ar = out[-(1:2)])
}

run_decoder <- function(net, Z) predict_np(net$decoder, Z)

# Prior mean including the autoregressive terms:
# mu_t + sum_r gamma_r,t (z_{t-r} - mu_{t-r}).
ar_prior_mean <- function(mu, ar, z, lags) {
  out <- mu
  for (r in seq_along(ar)) {
    ok <- which(!is.na(lags[, r]))
    j <- lags[ok, r]
    out[ok, ] <- out[ok, ] + ar[[r]][ok, , drop = FALSE] *
      (z[j, , drop = FALSE] - mu[j, , drop = FALSE])
  }
  out
}

# Row indices of the r-th previous time point of the same series.
lag_index <- function(time, key, ar_order) {
  n <- length(time)
  rank <- match(time, sort(unique(time)))
  id <- paste(key, rank, sep = "\r")
  if (anyDuplicated(id)) {
    stop("Each series (location or group) can have only one observation ",
         "per time point. Use `group` to identify the series.", call. = FALSE)
  }
  matrix(vapply(seq_len(ar_order), function(r) {
    match(paste(key, rank - r, sep = "\r"), id)
  }, integer(n)), nrow = n)
}
