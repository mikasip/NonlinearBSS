#' Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder
#' (iVAE) using the input data.
#' @importFrom magrittr %>%
#' @importFrom Rdpack reprompt
#' @importFrom mathjaxr preview_rd
#' @importFrom graphics par
#' @importFrom stats model.matrix predict rnorm runif sd var
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
#' @param get_elbo Logical. If TRUE, the model returns the final ELBO value. Default is FALSE.
#' @param epochs A number of epochs for training.
#' @param batch_size A batch size for training.
#' @return An object of class iVAE.
#' \item{IC_unscaled}{Unscaled latent variable estimates.}
#' \item{IC}{Scaled latent variable estimates.}
#' \item{data_dim}{Dimension of the observed data.}
#' \item{sample_size}{Sample size of the training data.}
#' \item{aux_dim}{The dimension of auxiliary variable vector.}
#' \item{prior_mean_model}{A model, which outputs the means estimated by
#' the auxiliary function.}
#' \item{prior_log_var_model}{A model, which outputs the logarithmic variances 
#' estimated by the auxiliary function.}
#' \item{encoder}{The trained encoder.}
#' \item{decoder}{The trained decoder.}
#' \item{data_means}{The means of the
#' original data.}
#' \item{data_sds}{The standard deviations of the
#' original data.}
#' \item{IC_means}{The means of the
#' unscaled latent components.}
#' \item{IC_sds}{The standard deviations of the
#' unscaled latent components.}
#' \item{call_params}{The params for
#' the original iVAE method call.}
#' \item{elbo}{The ELBO value after training the model.}
#' \item{metrics}{Metrics of the training for each epoch.}
#' \item{call}{An output of how the method was called.}
#' \item{DNAME}{The name of the original data.}
#' 
#' @details The method constructs and trains an identifiable variational
#' autoencoder (iVAE) \insertCite{Khemakhem2020}{NonlinearBSS}
#' based on the given parameters.
#' iVAE is composed of an encoder \mjeqn{g}{ascii}, a decoder
#' \mjeqn{h}{ascii} and
#' an auxiliary function \mjeqn{w}{ascii}.
#' The encoder transforms the original data
#' \mjeqn{x}{ascii} into a mean and a variace
#' vectors.
#' The mean and the variance are then used to sample a latent representation by
#' using a reparametrization trick \insertCite{kingma2013auto}{NonlinearBSS}.
#' The decoder aims to transform the latent representation
#' \mjeqn{z}{ascii} back to the
#' original data. The auxiliary function estimates the mean and the
#' variance of the data based on the auxiliary
#' data.
#' The functions \mjeqn{g}{ascii}, \mjeqn{h}{ascii}
#' and \mjeqn{w}{ascii} are deep neural networks
#' with parameters
#' \mjeqn{\lambda=(\lambda_g, \lambda_h, \lambda_w)^\top}{ascii}.
#' The parameters are learned by minimizing the lower bound of the data
#' log-likelihood:
#' \mjdeqn{
#' \mathcal{L}(\theta |  x,  u) \geq
#' E_{q_{ \theta_{ g}}( z| x, u)} (
#' log \, p_{ \theta_{ h}}(x | z)  +
#' log \, p_{ \theta_{w}}(z | u) -
#' log \, q_{\theta_{g}}(z | x, u) ).
#' }{ascii}
#'
#' In the loss function, \mjeqn{ log \, p_{ \theta_{ h}}(x | z)}{ascii}
#' controls the reconstruction error, and have the distribution based on the
#' parameter \code{error_dist} where \code{gaussian} or
#' \code{laplace} are currently supported. The location of
#' the distribution is the original data and the scale parameter
#' is given by \code{error_dist_sigma}. The default value for
#' \code{error_dist_sigma} is \code{0.02}. By decreasing the value,
#' the loss function emphasizes more the reconstruction error and
#' by increasing, the reconstruction error has less weight.
#' The term \mjeqn{log \, p_{ \theta_{w}}(z | u) -
#' log \, q_{\theta_{g}}(z | x, u)}{ascii} tries to make the
#' distributions \mjeqn{p_{\theta_{w}}(z | u)}{ascii} and
#' \mjeqn{q_{\theta_{g}}(z | x, u)}{ascii} as similar as possible.
#' This term controls the disentangling and allows the method
#' to find the true latent components.
#' The parameter \code{source_dist} defines the distributions
#' \mjeqn{p_{ \theta_{w}}(z | u)}{ascii} and
#' \mjeqn{q_{\theta_{g}}(z | x, u)}{ascii}.
#' The distributions \code{gaussian} and \code{laplace} are currently
#' supported.
#' The parameters for \mjeqn{p_{ \theta_{w}}(z | u)}{ascii} are given by the
#' auxiliary function and the parameters for
#' \mjeqn{q_{\theta_{g}}(z | x, u)}{ascii} are given by the encoder.
#'
#' The method is identifiable, meaning that it finds the true latent
#' representation in the limit of infinite data, if some conditions are
#' satisfied. Loosely speaking, in case of gaussian and laplace distributions,
#' the conditions are that the mixing
#' function is differentiable and that the variance (or scale)
#' of the latent sources are varying based on the auxiliary variable.
#' The exact identifiability results are given in
#' \insertCite{Khemakhem2020}{NonlinearBSS}.
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'   for (seg in 1:n_segments) {
#'     start_ind <- (seg - 1) * n_per_segment + 1
#'     end_ind <- seg * n_per_segment
#'     aux_data[start_ind:end_ind, seg] <- 1
#'     latent_data[start_ind:end_ind, i] <- rnorm(
#'       n_per_segment,
#'       runif(1, -5, 5), runif(1, 0.1, 5)
#'     )
#'   }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#'
#' # For better performance, increase the number of epochs
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' cormat <- cor(res$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#' @export
iVAE <- function(data, aux_data, latent_dim, hidden_units = c(128, 128, 128), aux_hidden_units = c(128, 128, 128),
                 activation = "leaky_relu", source_dist = "gaussian", validation_split = 0, error_dist = "gaussian",
                 error_dist_sigma = 0.02, optimizer = NULL, lr_start = 0.001, lr_end = 0.0001,
                 get_elbo = TRUE, steps = 10000, seed = NULL, epochs, batch_size) {
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

  mask <- (!is.na(data)) * 1L
  data_means <- colMeans(data, na.rm = TRUE)
  data_sds <- apply(data, 2, function(col) { sd(col, na.rm = TRUE) })
  data_cent <- sweep(data, 2, data_means, "-")
  data_scaled <- sweep(data_cent, 2, data_sds, "/")
  data_scaled[which(mask == 0)] <- 0

  if (!is.null(seed)) {
    tensorflow::tf$keras$utils$set_random_seed(as.integer(seed))
  }
  n <- as.integer(dim(data)[1])
  p <- as.integer(dim(data)[2])
  dim_aux <- as.integer(dim(aux_data)[2])
  if (n != dim(aux_data)[1]) {
    stop("Observed data and auxiliary data must have same sample size")
  }

  mask_input <- keras3::layer_input(p)
  input_prior <- keras3::layer_input(dim_aux)
  prior_v <- input_prior
  for (n_units in aux_hidden_units) {
    prior_v <- prior_v %>%
      keras3::layer_dense(units = n_units, activation = activation)
  }
  prior_mean <- prior_v %>% keras3::layer_dense(latent_dim)
  prior_log_var <- prior_v %>% keras3::layer_dense(latent_dim)
  prior_v <- keras3::layer_concatenate(list(prior_mean, prior_log_var))
  prior_mean_model <- keras3::keras_model(input_prior, prior_mean)
  prior_log_var_model <- keras3::keras_model(input_prior, prior_log_var)

  input_data <- keras3::layer_input(p)
  input_aux <- keras3::layer_input(dim_aux)
  input <- keras3::layer_concatenate(list(input_data, input_aux))
  submodel <- input
  for (n_units in hidden_units) {
    submodel <- submodel %>%
      keras3::layer_dense(units = n_units, activation = activation)
  }
  z_mean <- submodel %>% keras3::layer_dense(latent_dim)
  z_log_var <- submodel %>% keras3::layer_dense(latent_dim)
  z_mean_and_var <- keras3::layer_concatenate(list(z_mean, z_log_var))
  encoder <- keras3::keras_model(list(input_data, input_aux), z_mean)
  z_log_var_model <- keras3::keras_model(list(input_data, input_aux), z_log_var)

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
  final_output <- keras3::layer_concatenate(list(x_decoded_mean, z, z_mean_and_var, prior_v, mask_input))

  vae <- keras3::keras_model(list(input_data, input_aux, input_prior, mask_input), final_output)
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
    optimizer <- tensorflow::tf$keras$optimizers$Adam(learning_rate = tensorflow::tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2))
  }

  metric_reconst_accuracy <- custom_metric("metric_reconst_accuracy", function(x, res) {
    x_mean <- res[, 1:p]
    mask <- res[, (p + 5 * latent_dim + 1):(2 * p + 5 * latent_dim)]
    log_px_z_unreduced <- error_log_pdf(x, x_mean, tensorflow::tf$constant(error_dist_sigma, "float32"), reduce = FALSE)
    masked_log_px_z <- log_px_z_unreduced * mask
    log_px_z <- tensorflow::tf$reduce_sum(masked_log_px_z, axis = -1L)
    return(tensorflow::tf$reduce_mean(log_px_z, -1L))
  })

  metric_kl_vae <- custom_metric("metric_kl_vae", function(x, res) {
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

  hist <- vae %>% keras3::fit(list(data_scaled, aux_data, aux_data, mask), data_scaled, validation_split = validation_split, shuffle = TRUE, batch_size = batch_size, epochs = epochs)
  IC_estimates <- predict(encoder, list(data_scaled, aux_data))
  obs_estimates <- predict(decoder, IC_estimates)
  if (get_elbo) {
    print("Calculating ELBO...")
    IC_log_vars <- predict(z_log_var_model, list(data_scaled, aux_data))
    prior_means <- predict(prior_mean_model, aux_data)
    prior_log_vars <- predict(prior_log_var_model, aux_data)
    log_px_z <- error_log_pdf(tensorflow::tf$constant(data_scaled, "float32"), tensorflow::tf$cast(obs_estimates, "float32"), tensorflow::tf$constant(error_dist_sigma, "float32"))
    log_qz_xu <- source_log_pdf(tensorflow::tf$cast(IC_estimates, "float32"), tensorflow::tf$cast(IC_estimates, "float32"), tensorflow::tf$math$exp(tensorflow::tf$cast(IC_log_vars, "float32")))
    log_pz_u <- source_log_pdf(tensorflow::tf$cast(IC_estimates, "float32"), tensorflow::tf$cast(prior_means, "float32"), tensorflow::tf$math$exp(tensorflow::tf$cast(prior_log_vars, "float32")))
    elbo <- tensorflow::tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L)
    elbo <- as.numeric(elbo)
  } else elbo <- NULL
  IC_means <- colMeans(IC_estimates)
  IC_sds <- apply(IC_estimates, 2, sd)
  IC_estimates_cent <- sweep(IC_estimates, 2, IC_means, "-")
  IC_estimates_scaled <- sweep(IC_estimates_cent, 2, IC_sds, "/")

  iVAE_object <- list(
    IC_unscaled = IC_estimates, IC = IC_estimates_scaled, data_dim = p,
    sample_size = n, prior_mean_model = prior_mean_model, prior_log_var_model = prior_log_var_model,
    aux_dim = dim_aux, encoder = encoder, decoder = decoder, data_means = data_means,
    data_sds = data_sds, IC_means = IC_means, IC_sds = IC_sds,
    call_params = call_params, elbo = elbo, metrics = hist, call = deparse(sys.call()),
    DNAME = paste(deparse(substitute(data)))
  )

  class(iVAE_object) <- "iVAE"
  return(iVAE_object)
}

#' Plotting an Object of Class iVAE
#' @description \code{plot} method for the class \code{iVAE}.
#' @param x Object of class \code{iVAE}
#' @param IC_inds A vector providing indices of ICs which are plotted.
#' @param sample_inds A vector providing sample indices which are plotted.
#' @param unscaled A boolean determining if the unscaled ICs are plotted
#' or not.
#' @param type linetype for plot.default function.
#' @param xlab xlab for the bottom most plot.
#' @param ylabs A vector providing ylab params for the plots.
#' @param colors A vector providing colors for the plots.
#' @param oma A \code{oma} parameter for the function \code{par}.
#' See \code{\link[graphics]{par}}
#' @param mar A \code{mar} parameter for the function \code{par}.
#' See \code{\link[graphics]{par}}
#' @param ... Further parameters for all plot.default methods.
#' @returns
#' No return value.
#' @details
#' Plots the components provided by the parameter \code{IC_inds}. A sample
#' subset can be provided by parameter \code{sample_inds}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
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
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' plot(res)
#' @export
#' @author Mika Sipilä
#' @method plot iVAE
plot.iVAE <- function(
    x, IC_inds = 1:x$call_params$latent_dim,
    sample_inds = 1:dim(x$IC)[1], unscaled = FALSE, type = "l",
    xlab = "", ylabs = c(), colors = c(), oma = c(1, 1, 0, 0),
    mar = c(2, 2, 1, 1), ...) {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  if (unscaled) {
    ICs <- x$IC_unscaled[sample_inds, IC_inds]
  } else {
    ICs <- x$IC[sample_inds, IC_inds]
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

#' Printing an Object of Class iVAE
#' @description \code{print} method for the class \code{iVAE}.
#' @param x Object of class \code{iVAE}
#' @param ... Further parameters for \code{print.default}.
#' @returns
#' No return value.
#' @details
#' Prints details about the object of class \code{iVAE}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
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
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' print(res)
#' @export
#' @author Mika Sipilä
#' @method print iVAE
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

#' Summarize an Object of Class iVAE
#' @description \code{summary} method for the class \code{iVAE}.
#' @param object Object of class \code{iVAE}
#' @param ... Further parameters for \code{summary.default}.
#' @returns
#' No return value.
#' @details
#' Summarizes the object of class \code{iVAE}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
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
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' summary(res)
#' @export
#' @author Mika Sipilä
#' @method summary iVAE
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

#' Predict Method for an Object of Class iVAE
#' @description \code{predict} method for the class \code{iVAE}.
#' @param object Object of class \code{iVAE}
#' @param newdata A matrix containing the new data. It can contain
#' new observations or new latent component. Choose the parameter
#' \code{IC_to_data} accordingly.
#' @param aux_data A matrix containing auxiliary data for new observations.
#' Should not be provided if \code{IC_to_data} is \code{TRUE}.
#' @param IC_to_data A boolean defining if the prediction is done from
#' latent components to observations or from observations to latent
#' components. If \code{IC_to_data = TRUE}, \code{newdata} should contain
#' new latent components. Otherwise it should contain the new observations.
#' @param ... Further parameters for \code{predict.default}.
#' @returns
#' A matrix containing the predictions.
#' @details
#' Summarizes the object of class \code{iVAE}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
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
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' new_data <- matrix(rnorm(p * 2), nrow = 2)
#' new_aux_data <- rbind(c(1, rep(0, n_segments - 1)), c(1, rep(0, n_segments - 1)))
#' pred_ICs <- predict(res, new_data, new_aux_data)
#' new_ICs <- matrix(rnorm(p * 2), nrow = 2)
#' pred_obs <- predict(res, new_ICs, IC_to_data = TRUE)
#' @export
#' @author Mika Sipilä
#' @method predict iVAE
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
      ), "of the keras3::fitted model."
    ))
  }
  if (!IC_to_data) {
    aux_dim <- dim(aux_data)[2]
    aux_n <- dim(aux_data)[1]
    if (aux_dim != object$aux_dim) {
      stop("The dimension of the auxiliary data does not match
      the dimension of the auxiliary of the keras3::fitted model.")
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

#' Save iVAE Object with Trained Tensorflow Models
#' @description Saves \code{iVAE} object including the trained
#' tensorflow models.
#' @inheritDotParams base::save -list
#' @import keras3
#' @param object Object of class \code{iVAE}.
#' @param tf_model_dir Directory, where the tensorflow.
#' models should be saved.
#' @param file Filename for saving the \code{iVAE} object.
#' @param ... Further arguments for base::save function.
#' @return
#' No return value.
#' @details
#' The function saves iVAE object correctly by saving also the trained
#' tensorflow models. In a new session, the trained model is available
#' by loading the \code{iVAE} object using the function
#' \code{\link{load_with_tf}}.
#' @seealso
#' \code{\link{load_with_tf}}
#' @author Mika Sipilä
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
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
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' save_with_tf(res, "res_dir", "res_obj.RData")
#' loaded_obj <- load_with_tf("res_obj.RData")
#' new_ICs <- matrix(rnorm(p * 2), nrow = 2)
#' pred_obs <- predict(loaded_obj, new_ICs, IC_to_data = TRUE)
#' @export
save_with_tf <- function(object, tf_model_dir, file, ...) {
  tf_model_dir <- find_unique_name(tf_model_dir)
  if (!dir.exists(tf_model_dir)) dir.create(tf_model_dir)
  object$tf_model_dir <- tf_model_dir
  keras3::save_model(object$encoder, paste0(tf_model_dir, "/encoder.keras"))
  keras3::save_model(object$decoder, paste0(tf_model_dir, "/decoder.keras"))
  keras3::save_model(object$prior_mean_model, paste0(tf_model_dir, "/prior_mean_model.keras"))
  if ("iVAEar" %in% class(object)) {
    keras3::save_model(object$prior_ar_model, paste0(tf_model_dir, "/prior_ar_model.keras"))
  }
  save(object, file = file, ...)
  print("The model is saved successfully. Use the method load_with_tf to load the model correctly.")
}


#' Load iVAE Object with Trained Tensorflow Models
#' @description Loads \code{iVAE} object including the trained
#' tensorflow models.
#' @import keras3
#' @param file Filename of the saved \code{iVAE} object.
#' @return
#' A loaded \code{iVAE} object with restored trained Tensorflow models.
#' @details
#' The function loads iVAE object correctly by loading also the trained
#' tensorflow models.
#' @seealso
#' \code{\link{save_with_tf}}
#' @author Mika Sipilä
#' @inherit save_with_tf examples
#' @export
load_with_tf <- function(file) {
  obj_name <- load(file)
  object <- get(obj_name)
  object$encoder <- keras3::load_model(paste0(object$tf_model_dir, "/encoder.keras"))
  object$decoder <- keras3::load_model(paste0(object$tf_model_dir, "/decoder.keras"))
  object$prior_mean_model <- keras3::load_model(paste0(object$tf_model_dir, "/prior_mean_model.keras"))
  if ("iVAEar" %in% class(object)) {
    object$prior_ar_model <-  keras3::load_model(paste0(object$tf_model_dir, "/prior_ar_model.keras"))
  }
  return(object)
}