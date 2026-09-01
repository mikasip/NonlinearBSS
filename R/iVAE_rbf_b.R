
#' Identifiable Variational Autoencoder with Batch-wise Radial Basis Auxiliary Data
#'
#' @description Memory-efficient counterpart of \code{\link{iVAE}} that builds the
#' spatial/temporal radial basis function (RBF) auxiliary features batch-wise, inside
#' the Keras computational graph, from raw \code{spatial_locations} and \code{time_points}
#' instead of a precomputed n x d auxiliary matrix. Equivalent to \code{\link{iVAE}}
#' called with a precomputed RBF auxiliary matrix, up to small numerical differences.
#'
#' @param data A numeric matrix of observed data (n x p).
#' @param spatial_locations A numeric matrix of spatial locations (n x spatial_dim).
#' @param time_points A numeric vector of time points (length n).
#' @param latent_dim An integer specifying the latent dimension.
#' @param aux_extra Optional numeric matrix (n x m) of additional auxiliary covariates,
#' already expected to be raw (unscaled); it will be standardized internally.
#' @param spatial_dim Number of spatial dimensions. Default 2.
#' @param spatial_basis Vector of spatial resolution levels for the RBF construction.
#' @param temporal_basis Vector of temporal resolution levels for the RBF construction.
#' @param spatial_kernel Kernel used for spatial RBFs, \code{"gaussian"} or \code{"wendland"}.
#' @param week_component Logical, whether to include a day-of-week RBF component.
#' @param seasonal_period Optional seasonal period for temporal RBFs.
#' @param max_season Optional maximum number of seasons (see \code{\link{iVAEar_radial}}).
#' @param elevation Optional numeric vector of elevations.
#' @param elevation_basis Optional vector of elevation resolution levels. Required if
#' \code{elevation} is provided.
#' @param min_coords,max_coords Bounding coordinates used to normalize spatial locations
#' to [0, 1] (typically taken from the training data, see \code{\link{form_radial_params}}).
#' @param min_time_point,max_time_point Bounding time points used to normalize time.
#' @param hidden_units Vector of hidden units for encoder/decoder layers.
#' @param aux_hidden_units Vector of hidden units for the auxiliary (prior) network.
#' @param activation Activation function for hidden layers. Default \code{"leaky_relu"}.
#' @param source_dist Latent source distribution, \code{"gaussian"} or \code{"laplace"}.
#' @param validation_split Fraction of data used for validation.
#' @param error_dist Reconstruction error distribution, \code{"gaussian"} or \code{"laplace"}.
#' @param error_dist_sigma Standard deviation for the error distribution.
#' @param optimizer Optional Keras optimizer; defaults to Adam with polynomial decay.
#' @param lr_start,lr_end,steps Learning rate schedule parameters for the default optimizer.
#' @param add_mask_to_encoder Logical, whether to concatenate the missingness mask into the
#' encoder input when there is missing data. Default \code{TRUE}.
#' @param seed Optional integer random seed.
#' @param get_elbo Logical, whether to compute and return the final ELBO. Default \code{FALSE}.
#' @param epochs Number of training epochs.
#' @param batch_size Batch size for training.
#'
#' @return A fitted object of class \code{iVAE}, structurally equivalent to the output
#' of \code{\link{iVAE}}.
#'
#' @seealso \code{\link{iVAE}}, \code{\link{iVAE_radial_spatio_temporal_b}}
#' @author Mika Sipilä
#' @export
iVAE_rbf_b <- function(data, spatial_locations, time_points, latent_dim,
                        aux_extra = NULL,
                        spatial_dim = 2, spatial_basis = c(2, 9), temporal_basis = c(9, 17, 37),
                        spatial_kernel = "gaussian", week_component = FALSE,
                        seasonal_period = NULL, max_season = NULL,
                        elevation = NULL, elevation_basis = NULL,
                        min_coords, max_coords, min_time_point, max_time_point,
                        hidden_units = c(128, 128, 128), aux_hidden_units = c(128, 128, 128),
                        activation = "leaky_relu", source_dist = "gaussian", validation_split = 0,
                        error_dist = "gaussian", error_dist_sigma = 0.02, optimizer = NULL,
                        lr_start = 0.001, lr_end = 0.0001, steps = 10000,
                        add_mask_to_encoder = TRUE, seed = NULL, get_elbo = FALSE,
                        epochs, batch_size) {
 
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
    aux_hidden_units = aux_hidden_units, activation = activation,
    epochs = epochs, batch_size = batch_size, lr_start = lr_start,
    lr_end = lr_end, seed = seed, optimizer = optimizer
  )
 
  mask <- (!is.na(data)) * 1L
  n <- as.integer(dim(data)[1])
  p <- as.integer(dim(data)[2])
 
  data_means <- colMeans(data, na.rm = TRUE)
  data_sds <- apply(data, 2, function(col) { sd(col, na.rm = TRUE) })
  data_cent <- sweep(data, 2, data_means, "-")
  data_scaled <- sweep(data_cent, 2, data_sds, "/")
  data_scaled[which(mask == 0)] <- 0
 
  if (!is.null(seed)) {
    tensorflow::tf$keras$utils$set_random_seed(as.integer(seed))
  }
 
  # ---- raw spatial/temporal (/elevation/aux_extra) inputs ----
  spatial_dim_in <- as.integer(ncol(spatial_locations))
  spatial_input <- keras3::layer_input(shape = spatial_dim_in)
  temporal_input <- keras3::layer_input(shape = 1L)
  if (!is.null(elevation)) {
    elevation_input <- keras3::layer_input(shape = 1L)
  } else {
    elevation_input <- NULL
  }
  if (!is.null(aux_extra)) {
    aux_input_extra <- keras3::layer_input(shape = as.integer(ncol(aux_extra)))
  } else {
    aux_input_extra <- NULL
  }
 
  rbf_layer <- layer_radial_basis(
    spatial_basis = spatial_basis,
    temporal_basis = temporal_basis,
    min_coords = min_coords,
    max_coords = max_coords,
    min_time_point = min_time_point,
    max_time_point = max_time_point,
    spatial_kernel = spatial_kernel,
    week_component = week_component,
    seasonal_period = seasonal_period,
    max_season = max_season,
    elevation_basis = elevation_basis,
    min_elevation = if (!is.null(elevation)) min(elevation) else NULL,
    max_elevation = if (!is.null(elevation)) max(elevation) else NULL
  )
 
  if (!is.null(elevation_input)) {
    rbf_features <- rbf_layer(list(spatial_input, temporal_input, elevation_input))
  } else {
    rbf_features <- rbf_layer(list(spatial_input, temporal_input))
  }
 
  if (!is.null(aux_input_extra)) {
    aux_input <- keras3::layer_concatenate(list(rbf_features, aux_input_extra))
  } else {
    aux_input <- rbf_features
  }
 
  # inputs shared by the prior network (never includes the mask)
  rbf_inputs_for_prior <- list(spatial_input, temporal_input)
  if (!is.null(elevation_input)) rbf_inputs_for_prior <- append(rbf_inputs_for_prior, elevation_input)
  if (!is.null(aux_input_extra)) rbf_inputs_for_prior <- append(rbf_inputs_for_prior, aux_input_extra)
 
  # ---- prior (auxiliary) network ----
  prior_v <- aux_input
  for (n_units in aux_hidden_units) {
    prior_v <- prior_v %>% keras3::layer_dense(units = n_units, activation = activation)
  }
  prior_mean <- prior_v %>% keras3::layer_dense(units = latent_dim)
  prior_log_var <- prior_v %>% keras3::layer_dense(units = latent_dim)
  prior_v <- keras3::layer_concatenate(list(prior_mean, prior_log_var))
  prior_mean_model <- keras3::keras_model(rbf_inputs_for_prior, prior_mean)
  prior_log_var_model <- keras3::keras_model(rbf_inputs_for_prior, prior_log_var)
 
  # ---- encoder ----
  mask_input <- keras3::layer_input(p)
  input_data <- keras3::layer_input(p)
 
  use_mask_in_encoder <- !all(mask == 1) && add_mask_to_encoder
  if (use_mask_in_encoder) {
    encoder_input_list <- list(input_data, aux_input, mask_input)
  } else {
    encoder_input_list <- list(input_data, aux_input)
  }
  submodel <- keras3::layer_concatenate(encoder_input_list)
  for (n_units in hidden_units) {
    submodel <- submodel %>% keras3::layer_dense(units = n_units, activation = activation)
  }
  z_mean <- submodel %>% keras3::layer_dense(units = latent_dim)
  z_log_var <- submodel %>% keras3::layer_dense(units = latent_dim)
  z_mean_and_var <- keras3::layer_concatenate(list(z_mean, z_log_var))
 
  # canonical raw-input order used consistently for build / fit / predict:
  # data, spatial, temporal, [elevation], [aux_extra], [mask]
  encoder_raw_inputs <- list(input_data, spatial_input, temporal_input)
  if (!is.null(elevation_input)) encoder_raw_inputs <- append(encoder_raw_inputs, elevation_input)
  if (!is.null(aux_input_extra)) encoder_raw_inputs <- append(encoder_raw_inputs, aux_input_extra)
  if (use_mask_in_encoder) encoder_raw_inputs <- append(encoder_raw_inputs, mask_input)
 
  encoder <- keras3::keras_model(encoder_raw_inputs, z_mean)
  z_log_var_model <- keras3::keras_model(encoder_raw_inputs, z_log_var)
 
  sampling_layer <- switch(source_dist,
    "gaussian" = sampling_gaussian(p = latent_dim),
    "laplace" = sampling_laplace(p = latent_dim)
  )
  z <- z_mean_and_var %>% sampling_layer()
 
  # ---- decoder ----
  x_decoded_mean <- z
  input_decoder <- keras3::layer_input(latent_dim)
  output_decoder <- input_decoder
  for (n_units in rev(hidden_units)) {
    dense_layer <- keras3::layer_dense(units = n_units, activation = activation)
    x_decoded_mean <- x_decoded_mean %>% dense_layer()
    output_decoder <- output_decoder %>% dense_layer()
  }
  out_layer <- keras3::layer_dense(units = p)
  x_decoded_mean <- x_decoded_mean %>% out_layer()
  output_decoder <- output_decoder %>% out_layer()
  decoder <- keras3::keras_model(input_decoder, output_decoder)
 
  final_output <- keras3::layer_concatenate(list(x_decoded_mean, z, z_mean_and_var, prior_v, mask_input))
 
  # vae (training) inputs: same canonical order, mask is always present here
  # (the loss always needs it for masked reconstruction, independent of add_mask_to_encoder)
  vae_inputs <- list(input_data, spatial_input, temporal_input)
  if (!is.null(elevation_input)) vae_inputs <- append(vae_inputs, elevation_input)
  if (!is.null(aux_input_extra)) vae_inputs <- append(vae_inputs, aux_input_extra)
  vae_inputs <- append(vae_inputs, mask_input)
 
  vae <- keras3::keras_model(vae_inputs, final_output)
 
  vae_loss <- function(x, res) {
    x_mean <- res[, 1:p]
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    prior_mean_v <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_log_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    mask <- res[, (p + 5 * latent_dim + 1):(2 * p + 5 * latent_dim)]
    log_px_z_unreduced <- error_log_pdf(x, x_mean, tensorflow::tf$constant(error_dist_sigma, "float32"), reduce = FALSE)
    masked_log_px_z <- log_px_z_unreduced * mask
    log_px_z <- tensorflow::tf$reduce_sum(masked_log_px_z, axis = -1L)
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tensorflow::tf$math$exp(z_logvar))
    log_pz_u <- source_log_pdf(z_sample, prior_mean_v, tensorflow::tf$math$exp(prior_log_v))
    return(-tensorflow::tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L))
  }
 
  if (is.null(optimizer)) {
    optimizer <- tensorflow::tf$keras$optimizers$Adam(
      learning_rate = tensorflow::tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2)
    )
  }
 
  metric_reconst_accuracy <- keras3::custom_metric("metric_reconst_accuracy", function(x, res) {
    x_mean <- res[, 1:p]
    mask <- res[, (p + 5 * latent_dim + 1):(2 * p + 5 * latent_dim)]
    log_px_z_unreduced <- error_log_pdf(x, x_mean, tensorflow::tf$constant(error_dist_sigma, "float32"), reduce = FALSE)
    masked_log_px_z <- log_px_z_unreduced * mask
    log_px_z <- tensorflow::tf$reduce_sum(masked_log_px_z, axis = -1L)
    return(tensorflow::tf$reduce_mean(log_px_z, -1L))
  })
 
  metric_kl_vae <- keras3::custom_metric("metric_kl_vae", function(x, res) {
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    prior_mean_v <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_log_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tensorflow::tf$math$exp(z_logvar))
    log_pz_u <- source_log_pdf(z_sample, prior_mean_v, tensorflow::tf$math$exp(prior_log_v))
    return(-tensorflow::tf$reduce_mean((log_pz_u - log_qz_xu), -1L))
  })
 
  vae %>% keras3::compile(
    optimizer = optimizer,
    loss = vae_loss,
    metrics = list(metric_reconst_accuracy, metric_kl_vae)
  )
 
  # ---- fit: data in the same canonical order as vae_inputs ----
  inputs_to_fit <- list(data_scaled, spatial_locations, time_points)
  if (!is.null(elevation_input)) inputs_to_fit <- append(inputs_to_fit, list(elevation))
  if (!is.null(aux_input_extra)) inputs_to_fit <- append(inputs_to_fit, list(aux_extra))
  inputs_to_fit <- append(inputs_to_fit, list(mask))
 
  hist <- vae %>% keras3::fit(
    inputs_to_fit, data_scaled,
    validation_split = validation_split, shuffle = TRUE,
    batch_size = batch_size, epochs = epochs
  )
 
  # ---- predict: same canonical order as encoder_raw_inputs ----
  encoder_predict_inputs <- list(data_scaled, spatial_locations, time_points)
  if (!is.null(elevation_input)) encoder_predict_inputs <- append(encoder_predict_inputs, list(elevation))
  if (!is.null(aux_input_extra)) encoder_predict_inputs <- append(encoder_predict_inputs, list(aux_extra))
  if (use_mask_in_encoder) encoder_predict_inputs <- append(encoder_predict_inputs, list(mask))
 
  IC_estimates <- predict(encoder, encoder_predict_inputs)
  obs_estimates <- predict(decoder, IC_estimates)
 
  if (get_elbo) {
    print("Calculating ELBO...")
    IC_log_vars <- predict(z_log_var_model, encoder_predict_inputs)
 
    prior_predict_inputs <- list(spatial_locations, time_points)
    if (!is.null(elevation_input)) prior_predict_inputs <- append(prior_predict_inputs, list(elevation))
    if (!is.null(aux_input_extra)) prior_predict_inputs <- append(prior_predict_inputs, list(aux_extra))
 
    prior_means <- predict(prior_mean_model, prior_predict_inputs)
    prior_log_vars <- predict(prior_log_var_model, prior_predict_inputs)
 
    log_px_z <- error_log_pdf(
      tensorflow::tf$constant(data_scaled, "float32"),
      tensorflow::tf$cast(obs_estimates, "float32"),
      tensorflow::tf$constant(error_dist_sigma, "float32")
    )
    log_qz_xu <- source_log_pdf(
      tensorflow::tf$cast(IC_estimates, "float32"),
      tensorflow::tf$cast(IC_estimates, "float32"),
      tensorflow::tf$math$exp(tensorflow::tf$cast(IC_log_vars, "float32"))
    )
    log_pz_u <- source_log_pdf(
      tensorflow::tf$cast(IC_estimates, "float32"),
      tensorflow::tf$cast(prior_means, "float32"),
      tensorflow::tf$math$exp(tensorflow::tf$cast(prior_log_vars, "float32"))
    )
    elbo <- tensorflow::tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L)
    elbo <- as.numeric(elbo)
  } else {
    elbo <- NULL
  }
 
  IC_means <- colMeans(IC_estimates)
  IC_sds <- apply(IC_estimates, 2, sd)
  IC_estimates_cent <- sweep(IC_estimates, 2, IC_means, "-")
  IC_estimates_scaled <- sweep(IC_estimates_cent, 2, IC_sds, "/")
 
  iVAE_object <- list(
    IC_unscaled = IC_estimates, IC = IC_estimates_scaled, data_dim = p,
    sample_size = n, prior_mean_model = prior_mean_model, prior_log_var_model = prior_log_var_model,
    aux_dim = NULL, encoder = encoder, decoder = decoder, data_means = data_means,
    data_sds = data_sds, IC_means = IC_means, IC_sds = IC_sds,
    mask = mask, add_mask_to_encoder = add_mask_to_encoder,
    call_params = call_params, elbo = elbo, metrics = hist, call = deparse(sys.call()),
    DNAME = paste(deparse(substitute(data)))
  )
 
  class(iVAE_object) <- "iVAE"
  return(iVAE_object)
}