iVAE_ar1 <- function(data, aux_data, latent_dim, data_prev = NULL, test_data = NULL, test_data_aux = NULL, hidden_units = c(128, 128, 128), aux_hidden_units = c(128, 128, 128),
                     activation = "leaky_relu", source_dist = "gaussian", validation_split = 0, error_dist = "gaussian",
                     error_dist_sigma = 0.01, optimizer = NULL, lr_start = 0.01, lr_end = 0.0001,
                     steps = 10000, seed = NULL, get_prior_means = TRUE, true_data = NULL, epochs, batch_size) {
  source_dist <- match.arg(source_dist, c("gaussian", "laplace"))
  source_log_pdf <- switch(source_dist,
    "gaussian" = norm_log_pdf,
    "laplace" = laplace_log_pdf
  )
  error_dist <- match.arg(error_dist, c("gaussian", "laplace"))
  error_log_pdf <- switch(error_dist,
    "gaussian" = norm_log_pdf,
    "laplace" = laplace_log_pdf
  )
  call_params <- list(
    latent_dim = latent_dim, source_dist = source_dist, error_dist = error_dist,
    error_dist_sigma = error_dist_sigma, hidden_units = hidden_units,
    aux_hidden_units = aux_hidden_units, activation = "leaky_relu",
    epochs = epochs, batch_size = batch_size, lr_start = lr_start,
    lr_end = lr_end, seed = seed, optimizer = optimizer
  )

  n <- as.integer(dim(data)[1])
  p <- as.integer(dim(data)[2])

  data_means <- colMeans(data)
  data_sds <- apply(data, 2, sd)
  data_cent <- sweep(data, 2, data_means, "-")
  data_scaled <- sweep(data_cent, 2, data_sds, "/")
  if (!is.null(data_prev)) {
    data_prev <- sweep(data_prev, 2, data_means, "-")
    data_prev <- sweep(data_prev, 2, data_sds, "/")
  } else {
    data_prev <- rbind(rep(0, p), data_scaled[1:(n - 1), ])
  }
  test_data_scaled <- NULL
  if (!is.null(test_data)) {
    test_data_cent <- sweep(test_data, 2, data_means, "-")
    test_data_scaled <- sweep(test_data_cent, 2, data_sds, "/")
    test_data_prev <- rbind(rep(0, p), test_data_scaled[1:(n - 1), ])
  }

  if (!is.null(seed)) {
    tf$keras$utils$set_random_seed(as.integer(seed))
  }
  dim_aux <- as.integer(dim(aux_data)[2])
  if (n != dim(aux_data)[1]) {
    stop("Observed data and auxiliary data must have same sample size")
  }

  input_prior <- layer_input(dim_aux)
  prior_v <- input_prior
  for (n_units in aux_hidden_units) {
    prior_v <- prior_v %>%
      layer_dense(units = n_units, activation = activation)
  }
  prior_ar1 <- prior_v %>% layer_dense(units = latent_dim, activation = "tanh")
  prior_log_var <- prior_v %>% layer_dense(units = latent_dim)
  prior_v <- layer_concatenate(list(prior_ar1, prior_log_var))
  prior_ar1_model <- keras_model(input_prior, prior_ar1)
  prior_log_var_model <- keras_model(input_prior, prior_log_var)

  input_data <- layer_input(p)
  input_aux <- layer_input(dim_aux)
  input <- layer_concatenate(list(input_data, input_aux))
  input_prev_obs <- layer_input(p)
  input_prev <- layer_concatenate(list(input_prev_obs, input_aux))
  submodel <- input
  submodel_prev <- input_prev
  for (n_units in hidden_units) {
    new_layer <- layer_dense(units = n_units, activation = activation)
    submodel <- submodel %>% new_layer()
    submodel_prev <- submodel_prev %>% new_layer()
  }
  z_mean_layer <- layer_dense(units = latent_dim)
  z_mean <- submodel %>% z_mean_layer()
  z_log_var <- submodel %>% layer_dense(units = latent_dim)
  z_prev_mean <- submodel_prev %>% z_mean_layer()
  z_mean_and_var <- layer_concatenate(list(z_mean, z_log_var))
  encoder <- keras_model(list(input_data, input_aux), z_mean)
  z_log_var_model <- keras_model(list(input_data, input_aux), z_log_var)

  sampling_layer <- switch(source_dist,
    "gaussian" = sampling_gaussian(p = latent_dim),
    "laplace" = sampling_laplace(p = latent_dim)
  )
  z <- z_mean_and_var %>% sampling_layer()

  x_decoded_mean <- z
  input_decoder <- layer_input(latent_dim)
  output_decoder <- input_decoder
  for (n_units in rev(hidden_units)) {
    dense_layer <- layer_dense(units = n_units, activation = activation)
    x_decoded_mean <- x_decoded_mean %>%
      dense_layer()
    output_decoder <- output_decoder %>% dense_layer()
  }
  out_layer <- layer_dense(units = p)
  x_decoded_mean <- x_decoded_mean %>% out_layer()
  output_decoder <- output_decoder %>% out_layer()
  decoder <- keras_model(input_decoder, output_decoder)
  final_output <- layer_concatenate(list(x_decoded_mean, z, z_mean_and_var, z_prev_mean, prior_v))

  vae <- keras_model(list(input_data, input_prev_obs, input_aux, input_prior), final_output)
  vae_loss <- function(x, res) {
    x_mean <- res[, 1:p]
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    z_prev_mean <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_ar1 <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    prior_log_v <- res[, (p + 5 * latent_dim + 1):(p + 6 * latent_dim)]
    log_px_z <- error_log_pdf(x, x_mean, tf$constant(error_dist_sigma, "float32"))
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
    log_pz_u <- source_log_pdf(z_sample, tf$math$multiply(prior_ar1, z_prev_mean), tf$math$exp(prior_log_v))

    return(-tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L))
  }
  if (is.null(optimizer)) {
    optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2))
  }

  metric_reconst_accuracy <- custom_metric("metric_reconst_accuracy", function(x, res) {
    x_mean <- res[, 1:p]
    log_px_z <- error_log_pdf(x, x_mean, tf$constant(error_dist_sigma, "float32"))
    return(tf$reduce_mean(log_px_z, -1L))
  })

  metric_kl_vae <- custom_metric("metric_kl_vae", function(x, res) {
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    z_prev <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_ar1 <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    prior_log_v <- res[, (p + 5 * latent_dim + 1):(p + 6 * latent_dim)]
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
    log_pz_u <- source_log_pdf(z_sample, tf$math$multiply(prior_ar1, z_prev), tf$math$exp(prior_log_v))
    return(-tf$reduce_mean((log_pz_u - log_qz_xu), -1L))
  })

  mse_vae <- custom_metric("mse_vae", function(x, res) {
    x_mean <- res[, 1:p]
    return(metric_mean_squared_error(x, x_mean))
  })

  vae %>% compile(
    optimizer = optimizer,
    loss = vae_loss,
    metrics = list(metric_reconst_accuracy, metric_kl_vae, mse_vae)
  )
  validation_data <- NULL
  if (!is.null(test_data_scaled)) {
    validation_data <- list(list(test_data_scaled, test_data_prev, test_data_aux, test_data_aux), test_data_scaled)
  }
  MCCs <- numeric(epochs)
  if (!is.null(true_data)) {
    for (i in 1:epochs) {
      hist <- vae %>% fit(list(data_scaled, data_prev, aux_data, aux_data), data_scaled, validation_data = validation_data, validation_split = validation_split, shuffle = TRUE, batchsize = batchsize, epochs = 1, seed = seed)
      IC_estimates <- predict(encoder, list(data_scaled, aux_data))
      MCCs[i] <- absolute_mean_correlation(cor(IC_estimates, true_data))
    }
  } else {
    hist <- vae %>% fit(list(data_scaled, data_prev, aux_data, aux_data), data_scaled, validation_data = validation_data, validation_split = validation_split, shuffle = TRUE, batchsize = batchsize, epochs = epochs, seed = seed)
  }
  IC_estimates <- predict(encoder, list(data_scaled, aux_data))
  IC_log_vars <- predict(z_log_var_model, list(data_scaled, aux_data))
  IC_means <- colMeans(IC_estimates)
  IC_sds <- apply(IC_estimates, 2, sd)
  IC_estimates_cent <- sweep(IC_estimates, 2, IC_means, "-")
  IC_estimates_scaled <- sweep(IC_estimates_cent, 2, IC_sds, "/")
  IC_vars <- exp(IC_log_vars)
  IC_vars_scaled <- sweep(IC_vars, 2, IC_sds^2, "/")
  prior_means_scaled <- NULL
  prior_vars_scaled <- NULL
  # if (get_prior_means) {
  #   prior_ar1 <- predict(prior_ar1_model, diag(dim_aux))
  #   prior_log_vars <- predict(prior_log_var_model, diag(dim_aux))
  #   prior_means_cent <- sweep(prior_means, 2, IC_means, "-")
  #   prior_means_scaled <- sweep(prior_means_cent, 2, IC_sds, "/")
  #   prior_vars <- exp(prior_log_vars)
  #   prior_vars_scaled <- sweep(prior_vars, 2, IC_sds^2, "/")
  # }

  iVAE_object <- list(
    IC_unscaled = IC_estimates, IC = IC_estimates_scaled, IC_vars = IC_vars_scaled,
    prior_means = prior_means_scaled, prior_vars = prior_vars_scaled, data_dim = p,
    sample_size = n, prior_ar1_model = prior_ar1_model, call_params = call_params,
    aux_dim = dim_aux, encoder = encoder, decoder = decoder, data_means = data_means,
    data_sds = data_sds, IC_means = IC_means, IC_sds = IC_sds, MCCs = MCCs, call = deparse(sys.call()),
    DNAME = paste(deparse(substitute(data))), metrics = hist
  )

  class(iVAE_object) <- "iVAE"
  return(iVAE_object)
}
