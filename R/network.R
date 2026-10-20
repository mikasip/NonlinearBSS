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
#   has_missing       the data have missing values: the mask is a model input
#                     and the loss ignores the missing entries
#   use_mask          also use the mask as encoder input (only if has_missing
#                     and requested with `mask_encoder`)
#   seed
#
# The returned list contains the training model and the models used for
# prediction: encoder (mean and log variance), prior (mean, log variance and
# AR coefficients) and decoder.

build_network <- function(spec, step, jac = NULL) {
  L <- as.integer(spec$latent_dim)
  p <- as.integer(spec$p)
  R <- as.integer(spec$ar_order)
  has_aux <- spec$method != "VAE"
  is_graph <- spec$method == "GiVAE"
  G <- as.integer(spec$n_graphs)
  M <- as.integer(spec$max_neighbors)
  act <- spec$activation
  dense <- function(units, activation = NULL) {
    keras3::layer_dense(units = as.integer(units), activation = activation)
  }

  x_in <- keras3::layer_input(shape = p, name = "x")
  # The mask is a model input whenever there are missing values (loss), and is
  # given to the encoder only if `use_mask` (missing values + mask_encoder).
  has_mask <- isTRUE(spec$has_missing) || spec$use_mask
  mask_in <- if (has_mask) keras3::layer_input(shape = p, name = "mask") else NULL
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
    prior_mean_layer <- if (!is_graph) dense(L)
    prior_log_var_layer <- dense(L)
    prior_ar_layers <- lapply(seq_len(R), function(r) dense(L, "tanh"))
    if (is_graph) {
      # Coupling networks with their own hidden layers (separate from the
      # variance network). The coupling is COMPONENT-WISE: rho_l(u) in (0, 1)
      # is the total coupling of latent component l and beta_{g,l}(u) its split
      # between the graphs (softmax over graphs, per component). A coupling
      # shared by all components would make the prior invariant to rotations
      # of the latent space when the variances do not differ, so the
      # components could not be identified. With one graph the split is 1.
      coupling_hidden <- lapply(spec$aux_hidden_units, dense, activation = act)
      prior_rho_layer <- dense(L, "sigmoid")
      if (G > 1L) {
        prior_beta_layer <- dense(G * L)
        prior_beta_reshape <- keras3::layer_reshape(target_shape = c(G, L))
        prior_beta_softmax <- keras3::layer_activation_softmax(axis = -2L)
      } else {
        prior_beta_one <- keras3::layer_lambda(
          f = function(r) keras3::op_ones_like(keras3::op_expand_dims(r, -2L)),
          name = "beta_one")
      }
    }
    # coupling heads of the graph prior: rho (batch x L), beta (batch x G x L)
    coupling <- function(f) {
      hc <- f
      for (layer in coupling_hidden) hc <- layer(hc)
      rho <- prior_rho_layer(hc)
      beta <- if (G > 1L) {
        prior_beta_softmax(prior_beta_reshape(prior_beta_layer(hc)))
      } else {
        prior_beta_one(rho)
      }
      list(rho = rho, beta = beta)
    }
    prior <- function(f, coupling_only = FALSE) {
      if (is_graph && coupling_only) return(coupling(f))
      h <- f
      for (layer in prior_hidden) h <- layer(h)
      if (is_graph) {
        cp <- coupling(f)
        return(list(mean = NULL, log_var = prior_log_var_layer(h),
                    rho = cp$rho, beta = cp$beta, ar = list()))
      }
      list(mean = prior_mean_layer(h), log_var = prior_log_var_layer(h),
           ar = lapply(prior_ar_layers, function(layer) layer(h)))
    }
  }

  # Decoder h(z)
  dec_hidden <- lapply(rev(spec$hidden_units), dense, activation = act)
  dec_out <- dense(p)
  decode <- function(z) {
    for (layer in dec_hidden) z <- layer(z)
    dec_out(z)
  }

  # Training graph
  f0 <- if (has_aux) features(aux_in)
  enc <- encode(x_in, f0, if (spec$use_mask) mask_in else NULL)
  sampling <- nlbss_layer("sampling")(dist = spec$source_dist, seed = spec$seed)
  z <- sampling(list(enc$mean, enc$log_var))
  x_hat <- decode(z)

  # same order as network_inputs(): x, mask, aux
  inputs <- c(list(x_in), if (has_mask) list(mask_in), if (has_aux) list(aux_in))
  loss_mask <- if (has_mask) mask_in else keras3::op_ones_like(x_in)
  if (has_aux) {
    pr <- prior(f0)
    prior_mean <- pr$mean
    for (r in seq_len(R)) {
      x_lag <- keras3::layer_input(shape = p, name = paste0("x_lag", r))
      aux_lag <- keras3::layer_input(shape = as.integer(spec$n_aux),
                                     name = paste0("aux_lag", r))
      valid_lag <- keras3::layer_input(shape = 1L, name = paste0("valid_lag", r))
      lag_inputs <- list(x_lag, aux_lag, valid_lag)
      if (spec$use_mask) {
        mask_lag <- keras3::layer_input(shape = p, name = paste0("mask_lag", r))
        lag_inputs <- list(x_lag, mask_lag, aux_lag, valid_lag)
      }
      f_lag <- features(aux_lag)
      z_lag <- encode(x_lag, f_lag, if (spec$use_mask) mask_lag else NULL)$mean
      mean_lag <- prior(f_lag)$mean
      ar_term <- keras3::op_multiply(
        keras3::op_multiply(valid_lag, pr$ar[[r]]),
        keras3::op_subtract(z_lag, mean_lag)
      )
      prior_mean <- keras3::op_add(prior_mean, ar_term)
      inputs <- c(inputs, lag_inputs)
    }
    prior_log_var <- pr$log_var
    if (is_graph) {
      # neighbours of every graph, encoded with the shared encoder: the slots
      # (G * M of them) are a second input dimension and the dense layers act
      # on the last axis.
      S <- G * M
      nb_x <- keras3::layer_input(shape = c(S, p), name = "nb_x")
      nb_mask <- if (spec$use_mask) keras3::layer_input(shape = c(S, p), name = "nb_mask")
      nb_aux <- keras3::layer_input(shape = c(S, as.integer(spec$n_aux)), name = "nb_aux")
      nb_valid <- keras3::layer_input(shape = S, name = "nb_valid")
      z_nb <- encode(nb_x, features(nb_aux), nb_mask)$mean   # batch x S x L
      prior_mean <- graph_aggregate(z_nb, nb_valid, pr$rho, pr$beta, G, M, L,
                                    spec$rho_max)
      inputs <- c(inputs, list(nb_x), if (spec$use_mask) list(nb_mask),
                  list(nb_aux, nb_valid))
    }
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
  out <- elbo(list(x_in, x_hat, loss_mask, z, enc$mean, enc$log_var,
                   prior_mean, prior_log_var))
  jac_term <- NULL
  if (is_graph && !is.null(jac)) {
    # log-determinant (Jacobian) term of the SAR prior, added to the loss and
    # estimated by sampling a strongly connected component of the observation
    # graph at every step
    coef_fn <- function(a) {
      cp <- prior(features(a), coupling_only = TRUE)
      list(rho = cp$rho, beta = cp$beta)
    }
    jac_term <- jacobian_term_fn(jac, coef_fn, L, spec$rho_max)
  }
  train <- keras3::keras_model(inputs, out)

  # Prediction models
  enc_inputs <- c(list(x_in), if (spec$use_mask) list(mask_in),
                  if (has_aux) list(aux_in))
  encoder <- keras3::keras_model(enc_inputs, list(enc$mean, enc$log_var))
  z_in <- keras3::layer_input(shape = L)
  decoder <- keras3::keras_model(z_in, decode(z_in))
  prior_model <- if (has_aux && is_graph) {
    keras3::keras_model(aux_in, list(pr$log_var, pr$rho, pr$beta))
  } else if (has_aux) {
    keras3::keras_model(aux_in, c(list(pr$mean, pr$log_var), pr$ar))
  }
  list(train = train, encoder = encoder, decoder = decoder, prior = prior_model,
       method = spec$method, jacobian_term = jac_term)
}

# Graph prior mean from the encoder means of the neighbours:
#   rho_max * rho_l * sum_g beta_{g,l} * (mean of the valid neighbours in graph g)
# for every latent component l.
# z_nb: batch x (G * M) x L, valid: batch x (G * M), rho: batch x L,
# beta: batch x G x L. Slots of graph g are the columns (g - 1) * M + 1, ..., g * M.
#
# Done inside a Lambda layer: the ops then run on real tensors. Applying
# keras3::op_* with R scalars directly to symbolic KerasTensors can produce a
# tensor flagged `sparse = TRUE`, which fails in the ELBO layer.
graph_aggregate <- function(z_nb, valid, rho, beta, G, M, L, rho_max) {
  aggregate <- function(inputs) {
    z_nb <- inputs[[1]]
    valid <- inputs[[2]]
    rho <- inputs[[3]]
    beta <- inputs[[4]]
    z4 <- keras3::op_reshape(z_nb, c(-1L, G, M, L))
    v4 <- keras3::op_reshape(valid, c(-1L, G, M, 1L))
    num <- keras3::op_sum(keras3::op_multiply(z4, v4), axis = -2L)       # batch x G x L
    cnt <- keras3::op_maximum(keras3::op_sum(v4, axis = -2L), 1)         # batch x G x 1
    mean_g <- keras3::op_divide(num, cnt)
    agg <- keras3::op_sum(keras3::op_multiply(beta, mean_g), axis = -2L) # batch x L
    keras3::op_multiply(rho_max, keras3::op_multiply(rho, agg))
  }
  layer <- keras3::layer_lambda(f = aggregate, name = "graph_aggregate")
  layer(list(z_nb, valid, rho, beta))
}

# Inputs of the encoder model in the order used by build_network(): x, mask
# (only if the encoder uses it), aux.
encoder_inputs <- function(spec, X, A, mask) {
  inputs <- c(list(X), if (spec$use_mask) list(mask), if (spec$method != "VAE") list(A))
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
  if (identical(net$method, "GiVAE")) {
    # rho: n x L, beta: n x G x L (singleton dimensions may be dropped on the way)
    rho <- as.matrix(out[[2]])
    G <- as.integer(round(length(out[[3]]) / length(rho)))
    beta <- array(out[[3]], c(nrow(rho), G, ncol(rho)))
    return(list(mean = NULL, log_var = out[[1]], rho = rho, beta = beta))
  }
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