#' Trains an identifiable variational autoencoder (iVAE) using the input data.
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param aux_data A matrix with D columns and N rows
#' containing the auxiliary data.
#' @param latent_dim A latent dimension for iVAE
#' @param hidden_units K-dimensional vector giving the number of
#' hidden units for K layers in encoder and K layers in decoder.
#' @param aux_hidden_units K-dimensional vector giving the number of
#' hidden units for K layers in auxiliary function.
#' @param validation_split Proportion of data used for validation
#' @param activation Activation function for the hidden layers.
#' @param source_dist Distribution for the latent source.
#' Either "gaussian" or "laplace".
#' @param error_dist Distribution for the model error.
#' Either "gaussian" or "laplace".
#' @param error_dist_sigma A standard deviation for error_dist.
#' @param optimizer A keras optimizer for the tensorflow model.
#' A default is Adam optimizer with polynomial decay.
#' @param lr_start A starting learning rate for the default optimizer.
#' @param lr_end A ending learning rate for the default optimizer.
#' @param steps A number of learning steps between lr_start and lr_end
#' for the default optimizer.
#' @param seed Seed for the tensorflow model. Should be used instead of
#' set.seed(seed).
#' @param get_prior_means A boolean defining if the means provided by
#' the auxiliary function are returned.
#' @param true_data The true latent components. If provided, the mean
#' correlation coefficient is calculated for each epoch.
#' @param epochs A number of epochs for training.
#' @param batch_size A batch size for training.
#' @return An object of class VAE.
#' \item{IC_unscaled}{Unscaled latent components.}
#' \item{IC}{The latent component with
#' zero mean and unit variance.}
#' \item{data_dim}{The dimension of the original data.}
#' \item{metrics}{Metrics of the training for each epoch.}
#' \item{sample_size}{Sample size of
#' the data.}
#' \item{call_params}{The params for
#' the original VAE method call.}
#' \item{encoder}{The trained encoder.}
#' \item{decoder}{The trained decoder.}
#' \item{IC_means}{The means of the
#' unscaled latent components.}
#' \item{IC_means}{The standard devivations
#' of the unscaled latent components.}
#' \item{prior_means}{Means provided by auxiliary function.
#' Scaled to match the normalized latent components.}
#' \item{prior_vars}{Variances provided by auxiliary function.
#' Scaled to match the normalized latent components.}
#' \item{prior_mean_model}{A model, which outputs the means estimated by
#' the auxiliary function.}
#' \item{MCCs}{Mean correlation coefficients for each epoch.
#' Provided if true_data is not NULL.}
#' \item{call}{The of how the method was called.}
#' \item{DNAME}{The of the original data.}
#' @details The method constructs and trains an identifiable variational
#' autoencoder (iVAE) \insertRef{Khemakhem2020}{NonlinearBSS}
#' based on the given parameters.
#' iVAE is composed of an encoder, a decoder and an auxiliary function.
#' The encoder transforms the original data into a latent representation.
#' The decoder aims to transform the latent representation back to the
#' original data. The auxiliary function estimates the mean and the
#' variance of the data based on the auxiliary
#' data. The variational approximation is obtained by using a reparametrization
#' trick to sample a new value using the mean and the standard deviation
#' given by the encoder.
#' @references \insertAllCited{}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(NA, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'   for (seg in 1:n_segments) {
#'     start_ind <- (seg - 1) * n_per_segment + 1
#'     end_ind <- seg * n_per_segment
#'     latent_data[start_ind:end_ind, i] <- rnorm(
#'       n_per_segment,
#'       runif(1, -5, 5), runif(1, 0.1, 5)
#'     )
#'   }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 300, batch_size = 64)
#' cormat <- cor(res$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#' @export
iVAE <- function(
    data, aux_data, latent_dim, test_data = NULL,
    test_data_aux = NULL, hidden_units = c(128, 128, 128),
    aux_hidden_units = c(128, 128, 128), activation = "leaky_relu",
    source_dist = "gaussian", validation_split = 0,
    error_dist = "gaussian", error_dist_sigma = 0.02,
    optimizer = NULL, lr_start = 0.001, lr_end = 0.0001,
    steps = 10000, seed = NULL,
    get_prior_means = TRUE, true_data = NULL,
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
    aux_hidden_units = aux_hidden_units, activation = "leaky_relu",
    epochs = epochs, batch_size = batch_size, lr_start = lr_start,
    lr_end = lr_end, seed = seed, optimizer = optimizer
  )

  data_means <- colMeans(data)
  data_sds <- apply(data, 2, sd)
  data_cent <- sweep(data, 2, data_means, "-")
  data_scaled <- sweep(data_cent, 2, data_sds, "/")
  test_data_scaled <- NULL
  if (!is.null(test_data)) {
    test_data_cent <- sweep(test_data, 2, data_means, "-")
    test_data_scaled <- sweep(test_data_cent, 2, data_sds, "/")
  }

  if (!is.null(seed)) {
    tf$keras$utils$set_random_seed(as.integer(seed))
  }
  n <- as.integer(dim(data)[1])
  p <- as.integer(dim(data)[2])
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
  prior_mean <- prior_v %>% layer_dense(latent_dim)
  prior_log_var <- prior_v %>% layer_dense(latent_dim)
  prior_v <- layer_concatenate(list(prior_mean, prior_log_var))
  prior_mean_model <- keras_model(input_prior, prior_mean)
  prior_log_var_model <- keras_model(input_prior, prior_log_var)

  input_data <- layer_input(p)
  input_aux <- layer_input(dim_aux)
  input <- layer_concatenate(list(input_data, input_aux))
  submodel <- input
  for (n_units in hidden_units) {
    submodel <- submodel %>%
      layer_dense(units = n_units, activation = activation)
  }
  z_mean <- submodel %>% layer_dense(latent_dim)
  z_log_var <- submodel %>% layer_dense(latent_dim)
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
  final_output <- layer_concatenate(list(
    x_decoded_mean, z,
    z_mean_and_var, prior_v
  ))

  vae <- keras_model(list(input_data, input_aux, input_prior), final_output)
  vae_loss <- function(x, res) {
    x_mean <- res[, 1:p]
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    prior_log_v <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_mean_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    log_px_z <- error_log_pdf(
      x, x_mean,
      tf$constant(error_dist_sigma, "float32")
    )
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
    log_pz_u <- source_log_pdf(z_sample, prior_mean_v, tf$math$exp(prior_log_v))

    return(-tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L))
  }
  if (is.null(optimizer)) {
    optimizer <- tf$keras$optimizers$legacy$Adam(
      learning_rate =
        tf$keras$optimizers$schedules$PolynomialDecay(
          lr_start, steps,
          lr_end, 2
        )
    )
  }

  metric_reconst_accuracy <- custom_metric(
    "metric_reconst_accuracy",
    function(x, res) {
      x_mean <- res[, 1:p]
      log_px_z <- error_log_pdf(
        x, x_mean,
        tf$constant(error_dist_sigma, "float32")
      )
      return(tf$reduce_mean(log_px_z, -1L))
    }
  )

  metric_kl_vae <- custom_metric("metric_kl_vae", function(x, res) {
    z_sample <- res[, (1 + p):(p + latent_dim)]
    z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
    z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
    prior_log_v <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
    prior_mean_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
    log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
    log_pz_u <- source_log_pdf(z_sample, prior_mean_v, tf$math$exp(prior_log_v))
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
    validation_data <- list(list(
      test_data_scaled, test_data_aux,
      test_data_aux
    ), test_data_scaled)
  }
  MCCs <- numeric(epochs)
  if (!is.null(true_data)) {
    for (i in 1:epochs) {
      hist <- vae %>% fit(list(data_scaled, aux_data, aux_data),
        data_scaled,
        validation_data = validation_data,
        validation_split = validation_split, shuffle = TRUE,
        batchsize = batchsize, epochs = 1, seed = seed
      )
      IC_estimates <- predict(encoder, list(data_scaled, aux_data))
      MCCs[i] <- absolute_mean_correlation(cor(IC_estimates, true_data))
    }
  } else {
    hist <- vae %>% fit(list(data_scaled, aux_data, aux_data), data_scaled,
      validation_data = validation_data, validation_split = validation_split,
      shuffle = TRUE, batchsize = batchsize, epochs = epochs, seed = seed
    )
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
  if (get_prior_means) {
    prior_means <- predict(prior_mean_model, diag(dim_aux))
    prior_log_vars <- predict(prior_log_var_model, diag(dim_aux))
    prior_means_cent <- sweep(prior_means, 2, IC_means, "-")
    prior_means_scaled <- sweep(prior_means_cent, 2, IC_sds, "/")
    prior_vars <- exp(prior_log_vars)
    prior_vars_scaled <- sweep(prior_vars, 2, IC_sds^2, "/")
  }

  iVAE_object <- list(
    IC_unscaled = IC_estimates, IC = IC_estimates_scaled,
    IC_vars = IC_vars_scaled, prior_means = prior_means_scaled,
    prior_vars = prior_vars_scaled, data_dim = p, sample_size = n,
    prior_mean_model = prior_mean_model, call_params = call_params,
    aux_dim = dim_aux, encoder = encoder, decoder = decoder,
    data_means = data_means, data_sds = data_sds, IC_means = IC_means,
    IC_sds = IC_sds, MCCs = MCCs, call = deparse(sys.call()),
    D = paste(deparse(substitute(data))), metrics = hist
  )

  class(iVAE_object) <- "iVAE"
  return(iVAE_object)
}

#' Trains an identifiable variational autoencoder (iVAE) using the input data
#' and the segmented spatial domain as auxiliary data.
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param locations A matrix with spatial locations.
#' @param segment_sizes A vector providing sizes for segments.
#' The dimension should match the spatial dimenstion.
#' @param joint_segment_inds A vector indicating which segments.
#' are considered jointly. See more in details.
#' @param latent_dim A latent dimension for iVAE.
#' @param test_inds A vector giving the indices of the data, which
#' are used as a test data.
#' @param epochs A number of epochs for iVAE training.
#' @param batch_size A batch size for iVAE training.
#' @param ... Further parameters for \code{iVAE}.
#' @return
#' An object of class iVAESpatial, inherits from class iVAE.
#' Additionally, the object has a property
#' \code{spatial_dim} which gives the dimension of the given locations.
#' For more details, see [iVAE()].
#' @details
#' The method creates the auxiliary data as spatial segments based on
#' the given input parameters.
#' The vector \code{segment_sizes} defines the size of the segments for
#' each spatial dimension separately. The segmentation is then created
#' based on the \code{joint_segment_inds}, which defines dimensions are
#' considered jointly. For example, \code{joint_segment_inds = c(1, 1)}
#' for two dimensional spatial data, defines that the dimensions are
#' considered jointly. This means that the auxiliary variable is vector
#' giving the two dimensional segment that the observations belongs in.
#' For \code{joint_segment_inds = c(1, 2)}, the auxiliary variable would
#' consist of two vectors, first giving one dimensional segment that
#' the observation belongs in the first spatial dimension, and the second
#' giving one dimensional segment that the observation belongs in the
#' secoun spatial dimension. All dimensions are considered jointly as
#' default.
#'
#' After the segmentation, the method calls the function \code{iVAE}
#' using the created auxiliary variables.
#'
#' @references \insertAllCited{}
#' @seealso
#' [iVAE()]
#' @examples
#' # TODO
#' @export
iVAE_spatial <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim,
    test_inds = NULL, epochs, batch_size, ...) {
  n <- nrow(data)
  location_mins <- apply(locations, 2, min)
  locations_zero <- sweep(locations, 2, location_mins, "-")
  location_maxs <- apply(locations_zero, 2, max)
  aux_data <- matrix(nrow = n, ncol = 0)
  for (i in unique(joint_segment_inds)) {
    inds <- which(joint_segment_inds == i)
    labels <- rep(0, n)
    lab <- 1
    loop_dim <- function(j, sample_inds) {
      ind <- inds[j]
      seg_size <- segment_sizes[ind]
      seg_limits <- seq(0, (location_maxs[ind]), seg_size)
      for (coord in seg_limits) {
        cur_inds <- which(locations[sample_inds, ind] >= coord &
          locations[sample_inds, ind] < coord + seg_size)
        cur_sample_inds <- sample_inds[cur_inds]
        if (j == length(inds)) {
          labels[cur_sample_inds] <<- lab
          lab <<- lab + 1
        } else {
          loop_dim(j + 1, cur_sample_inds)
        }
      }
    }
    loop_dim(1, 1:n)
    labels <- as.numeric(as.factor(labels)) # To ensure that
    # empty segments are reduced
    aux_data <- cbind(aux_data, model.matrix(~ 0 + as.factor(labels)))
  }
  test_data <- NULL
  if (!is.null(test_inds)) {
    test_data <- data[test_inds, ]
    test_aux_data <- aux_data[test_inds, ]
  }
  resVAE <- iVAE(data, aux_data, latent_dim,
    test_data = test_data,
    test_data_aux = test_aux_data, epochs = epochs, batch_size = batch_size, ...
  )
  class(resVAE) <- c("iVAEspatial", class(resVAE))
  resVAE$spatial_dim <- dim(locations)[2]
  return(resVAE)
}

iVAE_spatial_ar <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), n_prev_obs = 1,
    latent_dim, epochs, batch_size, ...) {
  n <- nrow(data)
  p <- ncol(data)
  location_mins <- apply(locations, 2, min)
  locations_zero <- sweep(locations, 2, location_mins, "-")
  location_maxs <- apply(locations_zero, 2, max)
  aux_data <- matrix(nrow = n, ncol = 0)
  for (i in unique(joint_segment_inds)) {
    inds <- which(joint_segment_inds == i)
    labels <- rep(0, n)
    lab <- 1
    loop_dim <- function(j, sample_inds) {
      ind <- inds[j]
      seg_size <- segment_sizes[ind]
      seg_limits <- seq(0, (location_maxs[ind]), seg_size)
      for (coord in seg_limits) {
        cur_inds <- which(locations[sample_inds, ind] >= coord &
          locations[sample_inds, ind] < coord + seg_size)
        cur_sample_inds <- sample_inds[cur_inds]
        if (j == length(inds)) {
          labels[cur_sample_inds] <<- lab
          lab <<- lab + 1
        } else {
          loop_dim(j + 1, cur_sample_inds)
        }
      }
    }
    loop_dim(1, 1:n)
    labels <- as.numeric(as.factor(labels)) # To ensure
    # that empty segments are reduced
    aux_data <- cbind(aux_data, model.matrix(~ 0 + as.factor(labels)))
  }
  for (i in 1:n_prev_obs) {
    aux_i <- data
    for (j in 1:i) {
      aux_i <- rbind(c(rep(0, p)), aux_i[1:(n - 1), ])
    }
    aux_data <- cbind(aux_data, aux_i)
  }

  resVAE <- iVAE(data, aux_data, latent_dim,
    epochs = epochs, batch_size = batch_size, ...
  )
  class(resVAE) <- c("iVAEspatial", class(resVAE))
  resVAE$spatial_dim <- dim(locations)[2]
  return(resVAE)
}


gaussian_kernel <- function(d) {
  exp(-d^2)
}

wendland_kernel <- function(d) {
  (1 - d)^6 * (35 * d^2 + 18 * d + 3) / 3
}

iVAE_radial_basis <- function(
    data, locations, latent_dim,
    num_basis = c(10, 19, 37), kernel = "gaussian", epochs, batch_size, ...) {
  kernel <- match.arg(kernel, c("gaussian", "wendland"))
  spatial_dim <- dim(locations)[2]
  N <- dim(data)[1]
  min_coords <- apply(locations, 2, min)
  locations_new <- sweep(locations, 2, min_coords, "-")
  max_coords <- apply(locations_new, 2, max)
  locations_new <- sweep(locations_new, 2, max_coords, "/")
  knots_1d <- sapply(num_basis, FUN = function(i) {
    seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i)
  })
  phi_all <- matrix(0, ncol = 0, nrow = N)
  for (i in 1:length(num_basis)) {
    theta <- 1 / num_basis[i] * 2.5
    knot_list <- replicate(spatial_dim, knots_1d[[i]], simplify = FALSE)
    knots <- as.matrix(expand.grid(knot_list))
    phi <- cdist(locations_new, knots) / theta
    dist_leq_1 <- phi[which(phi <= 1)]
    dist_g_1_ind <- which(phi > 1)
    phi[which(phi <= 1)] <- switch(kernel,
      gaussian = gaussian_kernel(dist_leq_1),
      wendland = wendland_kernel(dist_leq_1)
    )
    phi[dist_g_1_ind] <- 0
    # ind_0 <- which(colSums(phi) == 0)
    # if (length(ind_0) > 0) {
    #  phi <- phi[, -which(colSums(phi) == 0)]
    # }
    phi_all <- cbind(phi_all, phi)
  }
  aux_data <- phi_all
  resVAE <- iVAE(data, aux_data, latent_dim,
    epochs = epochs,
    batch_size = batch_size, get_prior_means = FALSE, ...
  )
  class(resVAE) <- c("iVAEradial", class(resVAE))
  resVAE$min_coords <- min_coords
  resVAE$max_coords <- max_coords
  resVAE$num_basis <- num_basis
  # resVAE$phi_maxs <- phi_maxs
  resVAE$spatial_dim <- dim(locations)[2]
  return(resVAE)
}

iVAE_radial_spatio_temporal <- function(
    data, spatial_locations, time_points,
    latent_dim, spatial_dim = 2, spatial_basis = c(5, 9, 12),
    temporal_basis = c(10, 15, 45), spatial_kernel = "gaussian", epochs, batch_size, ...) {
  spatial_kernel <- match.arg(spatial_kernel, c("gaussian", "wendland"))
  N <- dim(data)[1]
  min_coords <- apply(spatial_locations, 2, min)
  locations_new <- sweep(spatial_locations, 2, min_coords, "-")
  max_coords <- apply(locations_new, 2, max)
  locations_new <- sweep(locations_new, 2, max_coords, "/")
  knots_1d <- sapply(spatial_basis, FUN = function(i) {
    seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i)
  })
  phi_all <- matrix(0, ncol = 0, nrow = N)
  kappa <- abs(temporal_basis[2] - temporal_basis[1])
  for (i in seq_along(spatial_basis)) {
    theta <- 1 / spatial_basis[i] * 2.5
    knot_list <- replicate(spatial_dim, knots_1d[[i]], simplify = FALSE)
    knots <- as.matrix(expand.grid(knot_list))
    phi <- cdist(locations_new, knots) / theta
    dist_leq_1 <- phi[which(phi <= 1)]
    dist_g_1_ind <- which(phi > 1)
    phi[which(phi <= 1)] <- switch(spatial_kernel,
      gaussian = gaussian_kernel(dist_leq_1),
      wendland = wendland_kernel(dist_leq_1)
    )
    phi[dist_g_1_ind] <- 0
    # ind_0 <- which(colSums(phi) == 0)
    # if (length(ind_0) > 0) {
    #  phi <- phi[, -which(colSums(phi) == 0)]
    # }
    phi_all <- cbind(phi_all, phi)
  }
  temp_knots <- unlist(sapply(temporal_basis, FUN = function(i) {
    seq(min(time_points), max(time_points), length.out = i)
  }))
  for (knot in temp_knots) {
    phi <- exp(-0.5 * (time_points - knot)^2 / kappa^2)
    phi_all <- cbind(phi_all, phi)
  }
  aux_data <- phi_all
  resVAE <- iVAE(data, aux_data, latent_dim,
    epochs = epochs,
    batch_size = batch_size, get_prior_means = FALSE, ...
  )
  class(resVAE) <- c("iVAEradial", class(resVAE))
  resVAE$min_coords <- min_coords
  resVAE$max_coords <- max_coords
  resVAE$spatial_basis <- spatial_basis
  # resVAE$phi_maxs <- phi_maxs
  resVAE$spatial_dim <- spatial_dim
  return(resVAE)
}


iVAE_coords <- function(data, locations, latent_dim, epochs, batch_size, ...) {
  n <- nrow(data)
  location_mins <- apply(locations, 2, min)
  locations_zero <- sweep(locations, 2, location_mins, "-")
  location_maxs <- apply(locations_zero, 2, max)
  locations_norm <- sweep(locations_zero, 2, location_maxs, "/")
  aux_data <- locations_norm

  resVAE <- iVAE(data, aux_data, latent_dim,
    epochs = epochs,
    batch_size = batch_size, ...
  )
  class(resVAE) <- c("iVAEcoords", class(resVAE))
  resVAE$spatial_dim <- dim(locations)[2]
  resVAE$location_mins <- location_mins
  resVAE$location_maxs <- location_maxs
  return(resVAE)
}

plot.iVAE <- function(
    obj, IC_inds = 1:obj$call_params$latent_dim,
    sample_inds = 1:dim(obj$ICs)[1], unscaled = FALSE, type = "l",
    xlab = "", ylabs = c(), colors = c(), oma = c(1, 1, 0, 0), mar = c(2, 2, 1, 1), ...) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (unscaled) {
    ICs <- obj$IC_unscaled[sample_inds, IC_inds]
  } else {
    ICs <- obj$IC[sample_inds, IC_inds]
  }
  p <- dim(ICs)[2]
  par(
    mfrow = c(p, 1),
    oma = oma,
    mar = mar,
    mgp = c(2, 0.5, 0),
    xpd = NA
  )
  for (i in 1:p) {
    if (length(ylabs) < i) ylabs[i] <- paste0("IC ", IC_inds[i])
  }
  for (i in 1:p) {
    if (is.null(colors)) {
      col <- 1
    } else {
      col <- ifelse(is.na(colors[i]), 1, colors[i])
    }
    plot(ICs[, i],
      type = type, xlab = ifelse(i == p, xlab, ""),
      ylab = ylabs[i], col = col, ...
    )
  }
}

print.iVAE <- function(x, ...) {
  cat("Call:", x$call)
  cat("\nData name:", x$DNAME)
  cat("\nDimension of observed data:", x$p)
  cat("\nLatent dimension:", x$call_params$latent_dim)
  cat("\nDimension of auxiliary data:", x$aux_dim)
  cat("\nSample size:", x$n)
  cat("\nSource density model: ", x$call_params$source_dist)
  cat("\nError density model: ", x$call_params$error_dist)
  cat("\nError sigma: ", x$call_params$error_dist_sigma)
}

summary.iVAE <- function(object, ...) {
  cat("Call:", object$call)
  cat("\nData name:", object$DNAME)
  cat("\nDimension of observed data:", object$p)
  cat("\nLatent dimension:", object$call_params$latent_dim)
  cat("\nDimension of auxiliary data:", object$aux_dim)
  cat("\nSample size:", object$n)
  cat("\nSource density model: ", object$call_params$source_dist)
  cat("\nError density model: ", object$call_params$error_dist)
  cat("\nError sigma: ", object$call_params$error_dist_sigma)
}

predict.iVAE <- function(
    object, newdata, aux_data = NULL,
    IC_to_data = FALSE, ...) {
  if ("numeric" %in% class(newdata)) {
    newdata <- t(as.matrix(newdata))
  } else {
    newdata <- as.matrix(newdata)
  }
  if (!IC_to_data) {
    if ("numeric" %in% class(aux_data)) {
      aux_data <- t(as.matrix(aux_data))
    } else {
      aux_data <- as.matrix(aux_data)
    }
  }
  dim_in <- ifelse(IC_to_data, object$call_params$latent_dim, object$data_dim)
  n <- dim(newdata)[1]
  p <- dim(newdata)[2]
  if (dim_in != p) {
    stop(paste0(
      "Dimension of newdata does not match the ",
      ifelse(IC_to_data, "latent dimension ",
        "dimension of the data "
      ), "of the fitted model."
    ))
  }
  if (!IC_to_data) {
    aux_dim <- dim(aux_data)[2]
    aux_n <- dim(aux_data)[1]
    if (aux_dim != object$aux_dim) {
      stop("The dimension of the auxiliary data does not match
      the dimension of the auxiliary of the fitted model.")
    }
    if (aux_n != n) {
      stop("The sample size must the same in newdata and aux_data.")
    }
  }
  if (IC_to_data) {
    newdata_unscaled <- sweep(newdata, 2, object$IC_sds, "*")
    newdata_unscaled_uncent <- sweep(newdata_unscaled, 2, object$IC_means, "+")
    obs_data_est <- object$decoder(newdata_unscaled_uncent)
    obs_data_est <- as.data.frame(as.matrix(obs_data_est))
    obs_data_est <- sweep(obs_data_est, 2, object$data_sds, "*")
    obs_data_est <- sweep(obs_data_est, 2, object$data_means, "+")
    names(obs_data_est) <- sapply(1:object$data_dim, FUN = function(x) {
      paste0("X", x)
    })
    return(obs_data_est)
  } else {
    newdata_cent <- sweep(newdata, 2, object$data_means, "-")
    newdata_scaled <- sweep(newdata_cent, 2, object$data_sds, "/")
    IC_est <- object$encoder(list(newdata_scaled, aux_data))
    IC_est <- as.matrix(IC_est)
    IC_est_cent <- sweep(IC_est, 2, object$IC_means, "-")
    IC_est_scaled <- sweep(IC_est_cent, 2, object$IC_sds, "/")
    IC_est_scaled <- as.data.frame(IC_est_scaled)
    names(IC_est_scaled) <- sapply(1:object$call_params$latent_dim,
      FUN = function(x) paste0("IC", x)
    )
    return(IC_est_scaled)
  }
}

predict.iVAEradial <- function(
    object, newdata, locations = NULL,
    IC_to_data = FALSE, ...) {
  if (IC_to_data) {
    res <- predict.iVAE(object, newdata, IC_to_data = TRUE)
    return(res)
  }
  locations_new <- sweep(locations, 2, object$min_coords, "-")
  locations_new <- sweep(locations_new, 2, object$max_coords, "/")
  knots_1d <- sapply(object$num_basis, FUN = function(i) {
    seq(0, 1, length.out = i)
  })
  phi_all <- matrix(0, ncol = 0, nrow = dim(newdata)[1])
  for (i in seq_along(object$num_basis)) {
    theta <- 1 / object$num_basis[i] * 2.5
    knot_list <- replicate(object$spatial_dim, knots_1d[[i]], simplify = FALSE)
    knots <- as.matrix(expand.grid(knot_list))
    phi <- cdist(locations_new, knots) / theta
    dist_leq_1 <- phi[which(phi <= 1)]
    dist_g_1_ind <- which(phi > 1)
    phi[which(phi <= 1)] <- (1 - dist_leq_1)^6 *
      (35 * dist_leq_1^2 + 18 * dist_leq_1 + 3) / 3
    phi[dist_g_1_ind] <- 0
    # ind_0 <- which(colSums(phi) == 0)
    # if (length(ind_0) > 0) {
    #  phi <- phi[, -which(colSums(phi) == 0)]
    # }
    phi_all <- cbind(phi_all, phi)
  }
  # phi_scaled <- sweep(phi_all, 2, object$phi_maxs, "/")
  aux_data <- phi_all
  res <- predict.iVAE(object, newdata, aux_data)
  return(res)
}

predict.iVAEcoords <- function(
    object, newdata,
    locations = NULL, IC_to_data = FALSE, ...) {
  if (IC_to_data) {
    res <- predict.iVAE(object, newdata, IC_to_data = TRUE)
    return(res)
  }
  locations_new <- sweep(locations, 2, object$location_mins, "-")
  locations_new <- sweep(locations_new, 2, object$location_maxs, "/")

  aux_data <- locations_new
  res <- predict.iVAE(object, newdata, aux_data)
  return(res)
}
