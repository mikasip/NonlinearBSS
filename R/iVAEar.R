#' Identifiable Variational Autoencoder with AR(R) Latent Structure
#'
#' @description \loadmathjax This function keras3::fits an Identifiable Variational Autoencoder (iVAE) model where the latent 
#' variables follow an AR(R) process, R being the autoregressive order.
#' It also supports handling auxiliary data and accommodates various 
#' choices for source and error distributions.
#'
#' @param data A numeric matrix of observed data (n x p), where n is the number of samples, 
#' and p is the number of features.
#' @param aux_data A numeric matrix of auxiliary data (n x m), where m is the number of auxiliary variables.
#' @param latent_dim An integer specifying the latent dimension.
#' @param prev_data_list A list of numeric matrices of previously observed data, 
#' used for constructing the AR(R) latent prior.
#' @param prev_aux_data_list A list of numeric matrices of auxiliary data for the previous observations.
#' @param hidden_units A numeric vector specifying the number of units in each hidden layer for the encoder and decoder.
#' @param aux_hidden_units A numeric vector specifying the number of units in each 
#' hidden layer in auxiliary function.
#' @param activation A string specifying the activation function to be used in hidden layers. Default is `"leaky_relu"`.
#' @param source_dist A character string specifying the distribution for the latent variables. Choices are `"gaussian"` (default) or `"laplace"`.
#' @param validation_split A numeric value specifying the fraction of training data to use for validation during model keras3::fitting. Default is 0.
#' @param error_dist A string specifying the distribution for the reconstruction error. Choices are `"gaussian"` (default) or `"laplace"`.
#' @param error_dist_sigma A numeric value specifying the standard deviation for the error distribution. Default is 0.01.
#' @param optimizer An optional Keras optimizer object. If NULL (default), an Adam optimizer with polynomial decay is used.
#' @param lr_start A numeric value for the initial learning rate. Default is 0.001.
#' @param lr_end A numeric value for the final learning rate. Default is 0.0001.
#' @param ar_order An autoregressive order used in iVAEar.
#' @param steps An integer specifying the number of optimization steps for polynomial learning rate decay. Default is 10,000.
#' @param seed An optional integer for setting the random seed to ensure reproducibility. Default is NULL.
#' @param get_elbo Logical. If TRUE, the model returns the final ELBO value. Default is FALSE.
#' @param epochs An integer specifying the number of training epochs.
#' @param batch_size An integer specifying the batch size for training.
#'
#' @details The iVAEar method extends spatio-temporal identifiable variational 
#' autoencoders (\code{\link{iVAE}}) with an autoregressive (AR) structure. It consists of:
#'
#' Encoder \mjeqn{\mathbf{g}(\mathbf{x}, \mathbf{u})}{ascii}: Maps observations 
#' \mjeqn{\mathbf{x}}{ascii} and auxiliary variables \mjeqn{\mathbf{u}}{ascii} to 
#' latent variables \mjeqn{\mathbf{z}}{ascii}.
#' 
#' Decoder \mjeqn{\mathbf{h}(\mathbf{x})}{ascii}: Reconstructs \mjeqn{\mathbf{x}}{ascii} 
#' from the latent representations.
#' 
#' Auxiliary Function \mjeqn{\mathbf{w}(\mathbf{u})}{ascii}: Estimates parameters of 
#' the autoregressive latent prior.
#' 
#' In particular, the auxiliary function gives spatio-temporal trend 
#' (\mjeqn{\mathbf{\mu}(\mathbf{s}, t)}{ascii}), variance (\mjeqn{\mathbf{\sigma}(\mathbf{s}, t)}{ascii}) 
#' and AR coefficient (\mjeqn{\mathbf{\gamma_r}(\mathbf{s}, t), r=1,\dots, W}{ascii}) functions. 
#' The parameter \mjeqn{W}{ascii} in the equations correspond to the parameter \code{ar_order} 
#' set by user, \mjeqn{\mathbf{s}}{ascii} is the spatial location and \mjeqn{t} is the 
#' temporal location. The latent components \mjeqn{z_1, \dots, z_P}{ascii} are 
#' assumed to be generated through the following autoregressive process:
#' 
#' \mjeqn{\mathbf{z}^t_i = \mu_i(\mathbf{s}, t) + \sum_{r=1}^{W} \gamma_{i,r}(\mathbf{s}, t) (\mathbf{z}_i^{t-r} - \mu_i(\mathbf{s}, t - r)) + \epsilon_i(\mathbf{s}, t)}{ascii}
#' 
#' where \mjeqn{\epsilon_i \sim \mathcal{N}(\mathbf{0}, \sigma(\mathbf{s}, t))}{ascii} is Gaussian noise.
#' 
#' The model optimizes the **evidence lower bound (ELBO):
#' 
#' \mjeqn{\mathcal{L} = \mathbb{E}_{q(\mathbf{z} | \mathbf{x}, \mathbf{u})} \left[ \log p(\mathbf{x} | \mathbf{z}) \right] - D_{\text{KL}}(q(\mathbf{z} | \mathbf{x}, \mathbf{u}) \| p(\mathbf{z} | \mathbf{u}))}{ascii}
#' 
#' where:
#' 
#' The first term maximizes reconstruction accuracy by ensuring 
#' \mjeqn{\mathbf{x}}{ascii} can be recovered from \mjeqn{\mathbf{z}}{ascii}.
#' 
#' The second term regularizes the latent space, enforcing an 
#' autoregressive prior structure through \mjeqn{\mathbf{w}(\mathbf{u})}{ascii}.
#' 
#' The framework is implemented using deep neural networks, optimizing 
#' via stochastic gradient descent. This approach ensures latent variables 
#' retain meaningful spatio-temporal dependencies, improving predictive 
#' performance in complex datasets.
#'
#' @return 
#' #' A keras3::fitted iVAEar object of class \code{iVAEar}, which inherits from class \code{\link{iVAE}}.
#' In addition the object has the following field:
#' \item{prior_ar_model}{A model, which outputs the estimated AR coefficients
#' by the auxiliary function.}
#' 
#' @examples
#' p <- 3
#' n_time <- 100
#' n_spat <- 50
#' coords_time <- cbind(
#'     rep(runif(n_spat), n_time), rep(runif(n_spat), n_time),
#'     rep(1:n_time, each = n_spat)
#' )
#' data_obj <- generate_nonstationary_spatio_temporal_data_by_segments(
#'     n_time,
#'     n_spat, p, 5, 10, coords_time
#' )
#' latent_data <- data_obj$data
#' # Generate artificial observed data by applying a nonlinear mixture
#' obs_data <- mix_data(latent_data, 2)
#'
#' # Increase the number of epochs for better performance.
#' resiVAE <- iVAEar_radial(
#'   data = obs_data, 
#'   spatial_locations = coords_time[, 1:2],
#'   time_points = coords_time[, 3],
#'   latent_dim = p, 
#'   n_s = n_spat,
#'   epochs = 1,
#'   batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' 
#' @export
iVAEar <- function(data, aux_data, latent_dim, prev_data_list, prev_aux_data_list, hidden_units = c(128, 128, 128), aux_hidden_units = c(128, 128, 128),
                     activation = "leaky_relu", source_dist = "gaussian", validation_split = 0, error_dist = "gaussian",
                     error_dist_sigma = 0.01, optimizer = NULL, lr_start = 0.001, lr_end = 0.0001, ar_order = 1,
                     steps = 10000, seed = NULL, get_elbo = TRUE, epochs, batch_size) {
  source_dist <- match.arg(source_dist, c("gaussian", "laplace"))
  source_log_pdf <- switch(source_dist,
    "gaussian" = norm_log_pdf,
    "laplace" = laplace_log_pdf,
  )
  error_dist <- match.arg(error_dist, c("gaussian", "laplace", "huber"))
  error_log_pdf <- switch(error_dist,
    "gaussian" = norm_log_pdf,
    "laplace" = laplace_log_pdf,
    "huber" = huber_loss,
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
  data_scaled[which(mask == 0)] <- rnorm(length(which(mask == 0)))
  
  for (i in seq_along(prev_data_list)) {
    prev_data_list[[i]][which(is.na(prev_data_list[[i]]))] <- 0
    prev_data_list[[i]] <- sweep(prev_data_list[[i]], 2, data_means, "-")
    prev_data_list[[i]] <- sweep(prev_data_list[[i]], 2, data_sds, "/")
  }

  if (!is.null(seed)) {
    tensorflow::tf$keras$utils$set_random_seed(as.integer(seed))
  }
  dim_aux <- as.integer(dim(aux_data)[2])
  if (n != dim(aux_data)[1]) {
    stop("Observed data and auxiliary data must have same sample size")
  }

  mask_input <- keras3::layer_input(p)
  aux_input <- keras3::layer_input(dim_aux)
  prior_v <- aux_input
  prev_aux_inputs <- list()
  prev_data_inputs <- list()
  prev_z <- list()
  prev_prior_means <- list()
  for (i in 1:ar_order) {
    input_aux_i <- keras3::layer_input(dim_aux)
    input_data_i <- keras3::layer_input(p)
    prev_aux_inputs <- append(prev_aux_inputs, input_aux_i)
    prev_prior_means <- append(prev_prior_means, input_aux_i)
    prev_data_inputs <- append(prev_data_inputs, input_data_i)
    prev_z <- append(prev_z, keras3::layer_concatenate(input_data_i, input_aux_i))
  }
  for (n_units in aux_hidden_units) {
    layer <- keras3::layer_dense(units = n_units, activation = activation)
    prior_v <- prior_v %>% layer()
    for (i in 1:ar_order) {
      prev_prior_means[[i]] <- prev_prior_means[[i]] %>% layer()
    }
  }
  prior_ar_coefs <- prior_v %>% keras3::layer_dense(units = latent_dim * ar_order, activation = "tanh")
  prior_ar_model <- keras3::keras_model(aux_input, prior_ar_coefs)
  
  prior_log_var <- prior_v %>% keras3::layer_dense(units = latent_dim)
  prior_mean_layer <- keras3::layer_dense(units = latent_dim)
  prior_mean <- prior_v %>% prior_mean_layer()
  prior_v <- keras3::layer_concatenate(list(prior_mean, prior_log_var, prior_ar_coefs))
  prior_log_var_model <- keras3::keras_model(aux_input, prior_log_var)
  prior_mean_model <- keras3::keras_model(aux_input, prior_mean)

  for (i in 1:ar_order) {
    prev_prior_means[[i]] <- prev_prior_means[[i]] %>% prior_mean_layer()  
  }

  input_data <- keras3::layer_input(p)
  input <- keras3::layer_concatenate(list(input_data, aux_input))
  submodel <- input
  for (n_units in hidden_units) {
    new_layer <- keras3::layer_dense(units = n_units, activation = activation)
    submodel <- submodel %>% new_layer()
    for (i in 1:ar_order) {
      prev_z[[i]] <- prev_z[[i]] %>% new_layer()
    }
  }
  z_mean_layer <- keras3::layer_dense(units = latent_dim)
  z_log_var_layer <- keras3::layer_dense(units = latent_dim)
  z_mean <- submodel %>% z_mean_layer()
  z_log_var <- submodel %>% z_log_var_layer()
  z_mean_and_var <- keras3::layer_concatenate(list(z_mean, z_log_var))
  encoder <- keras3::keras_model(list(input_data, aux_input), z_mean)
  z_log_var_model <- keras3::keras_model(list(input_data, aux_input), z_log_var)

  for (i in 1:ar_order) {
    prev_z[[i]] <- prev_z[[i]] %>% z_mean_layer()
  }

  sampling_layer <- switch(source_dist,
    "gaussian" = sampling_gaussian(p = latent_dim),
    "laplace" = sampling_laplace(p = latent_dim)
  )
  z <- z_mean_and_var %>% sampling_layer()

  x_decoded_mean <- z
  input_decoder <- keras3::layer_input(latent_dim)
  output_decoder <- input_decoder
  for (n_units in rev(hidden_units)) {
    dense_layer <- keras3::layer_dense(units = n_units, activation = activation)
    x_decoded_mean <- x_decoded_mean %>%
      dense_layer()
    output_decoder <- output_decoder %>% dense_layer()
  }
  out_layer <- keras3::layer_dense(units = p)
  x_decoded_mean <- x_decoded_mean %>% out_layer()
  output_decoder <- output_decoder %>% out_layer()
  decoder <- keras3::keras_model(input_decoder, output_decoder)
  output <- list(x_decoded_mean, z, z_mean_and_var, prior_v, mask_input)
  output <- append(output, prev_z)
  output <- append(output, prev_prior_means)
  final_output <- keras3::layer_concatenate(output)

  inputs <- list(input_data, aux_input, mask_input)
  inputs <- append(inputs, prev_data_inputs)
  inputs <- append(inputs, prev_aux_inputs)
  vae <- keras3::keras_model(inputs, final_output)
  vae_loss <- function(x, res) {
    x_mean <- res[, 1:p]
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    prior_mean <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_log_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    start_i <- p + 5 * latent_dim + 1
    end_i <- p + 5 * latent_dim + latent_dim * ar_order
    prior_ar <- res[, start_i:end_i]
    mask <- res[, (end_i + 1):(end_i + p)]
    prev_z <- res[, (end_i + p + 1):(end_i + p + ar_order * latent_dim)]
    prior_prev_mean <- res[, (end_i + p + ar_order * latent_dim + 1):(end_i + p + (2 * ar_order) * latent_dim)]
    ar_prev_mult <- tensorflow::tf$math$multiply(prior_ar, (prev_z - prior_prev_mean))
    prior_mean_final <- prior_mean + ar_prev_mult[, 1:latent_dim]
    if (ar_order > 1) {
      for (i in 2:ar_order) {
        prior_mean_final <- prior_mean_final + ar_prev_mult[, ((i - 1) * latent_dim + 1):(i * latent_dim)]
      }
    }
    log_px_z_unreduced <- error_log_pdf(x, x_mean, tensorflow::tf$constant(error_dist_sigma, "float32"), reduce = FALSE)
      
    masked_log_px_z <- log_px_z_unreduced * mask
    log_px_z <- tensorflow::tf$reduce_sum(masked_log_px_z, axis = -1L)
    
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tensorflow::tf$math$exp(z_logvar))
    
    log_pz_u <- source_log_pdf(z_sample, prior_mean_final, tensorflow::tf$math$exp(prior_log_v))
    
    return(-tensorflow::tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L))
  }
  if (is.null(optimizer)) {
    optimizer <- tensorflow::tf$keras$optimizers$Adam(learning_rate = tensorflow::tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2))
  }

  metric_reconst_accuracy <- custom_metric("metric_reconst_accuracy", function(x, res) {
    x_mean <- res[, 1:p]
    mask <- res[, (p + 5 * latent_dim + latent_dim * ar_order + 1):(2 * p + 5 * latent_dim + latent_dim * ar_order)]
    log_px_z_unreduced <- error_log_pdf(x, x_mean, tensorflow::tf$constant(error_dist_sigma, "float32"), reduce = FALSE)
    masked_log_px_z <- log_px_z_unreduced * mask
    log_px_z <- tensorflow::tf$reduce_sum(masked_log_px_z, axis = -1L)
    return(tensorflow::tf$reduce_mean(log_px_z, -1L))
  })

  vae %>% keras3::compile(
    optimizer = optimizer,
    loss = vae_loss,
    metrics = list(metric_reconst_accuracy)
  )

  inputs <- list(data_scaled, aux_data, mask)
  inputs <- append(inputs, prev_data_list)  
  inputs <- append(inputs, prev_aux_data_list)

  hist <- vae %>% keras3::fit(inputs, data_scaled, 
    validation_split = validation_split, shuffle = TRUE, 
    batch_size = batch_size, epochs = epochs)
  
  IC_estimates <- predict(encoder, list(data_scaled, aux_data))
  IC_log_vars <- predict(z_log_var_model, list(data_scaled, 
      aux_data))
  if (get_elbo) {
    print("Calculating ELBO...")
    obs_estimates <- predict(decoder, IC_estimates)
    prior_means <- predict(prior_mean_model, aux_data)
    prior_mean_ests <- prior_means
    for (i in 1:ar_order) {
      prior_means_prev <- predict(prior_mean_model, prev_aux_data_list[[1]])
      IC_estimates_prev <- predict(encoder,list(prev_data_list[[1]], prev_aux_data_list[[1]]))
      prior_log_vars <- predict(prior_log_var_model, aux_data)
      prior_ars <- predict(prior_ar_model, aux_data)
      prior_mean_ests <- prior_mean_ests + prior_ars[, ((i - 1) * latent_dim + 1):(i * latent_dim)] * (IC_estimates_prev - prior_means_prev)
    }
    log_px_z <- error_log_pdf(tensorflow::tf$constant(data_scaled, "float32"), 
        tensorflow::tf$cast(obs_estimates, "float32"), tensorflow::tf$constant(error_dist_sigma, 
            "float32"))
    log_qz_xu <- source_log_pdf(tensorflow::tf$cast(IC_estimates, "float32"), 
        tensorflow::tf$cast(IC_estimates, "float32"), tensorflow::tf$math$exp(tensorflow::tf$cast(IC_log_vars, 
            "float32")))
    log_pz_u <- source_log_pdf(tensorflow::tf$cast(IC_estimates, "float32"), 
        tensorflow::tf$cast(prior_mean_ests, "float32"), tensorflow::tf$math$exp(tensorflow::tf$cast(prior_log_vars, 
            "float32")))
    elbo <- tensorflow::tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L)
  } else elbo <- NULL
  
  IC_log_vars <- predict(z_log_var_model, list(data_scaled, aux_data))
  IC_means <- colMeans(IC_estimates)
  IC_sds <- apply(IC_estimates, 2, sd)
  IC_estimates_cent <- sweep(IC_estimates, 2, IC_means, "-")
  IC_estimates_scaled <- sweep(IC_estimates_cent, 2, IC_sds, "/")
  IC_vars <- exp(IC_log_vars)
  IC_vars_scaled <- sweep(IC_vars, 2, IC_sds^2, "/")

  iVAE_object <- list(
    IC_unscaled = IC_estimates, IC = IC_estimates_scaled, data_dim = p, ar_order = ar_order,
    sample_size = n, prior_ar_model = prior_ar_model, prior_mean_model = prior_mean_model,
    aux_dim = dim_aux, encoder = encoder, decoder = decoder, data_means = data_means,
    data_sds = data_sds, IC_means = IC_means, IC_sds = IC_sds, call_params = call_params, elbo = elbo, 
    metrics = hist, call = deparse(sys.call()), DNAME = paste(deparse(substitute(data)))
  )

  class(iVAE_object) <- c("iVAE", "iVAEar")
  return(iVAE_object)
}
