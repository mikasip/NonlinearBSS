#' Variational Autoencoder
#' @description Trains a variational autoencoder (VAE) using the input data.
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#' @param data A matrix with P columns and n rows
#' @param latent_dim A latent dimension for VAE
#' @param hidden_units K-dimensional vector giving the number of
#' hidden units for K layers in encoder and K layers in decoder.
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
#' \item{call}{The of how the method was called.}
#' \item{DNAME}{The of the original data.}
#' @details The method constructs and trains a variational autoencoder (VAE)
#' \insertCite{kingma2013auto}{NonlinearBSS} based on the given parameters.
#' VAE is composed of an encoder and a decoder. The encoder transforms the
#' original data into a latent representation. The decoder aims to
#' transform the latent representation back to the original data.
#' The variational approximation is obtained by using a reparametrization
#' trick to sample a new value using the mean and the standard deviation
#' given by the encoder.
#' @references \insertAllCited{}
#' @examples
#' p <- 3
#' n <- 1000
#' latent_data <- matrix(rnorm(p * n), ncol = p, nrow = n)
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' res <- VAE(mixed_data, p, hidden_units = c(128, 64, 32), 
#' lr_start = 0.0001, epochs = 1, batch_size = 64)
#' @export
CVAE <- function(data, aux_data, latent_dim, hidden_units = c(128, 128, 128), validation_split = 0,
                activation = "leaky_relu", source_dist = "gaussian", error_dist = "gaussian",
                error_dist_sigma = 0.01, optimizer = NULL, lr_start = 0.01, lr_end = 0.0001,
                steps = 10000, seed = NULL, epochs, batch_size) {
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
        activation = "leaky_relu",
        epochs = epochs, batch_size = batch_size, lr_start = lr_start,
        lr_end = lr_end, seed = seed, optimizer = optimizer
    )

    data_means <- colMeans(data)
    data_sds <- apply(data, 2, sd)
    data_cent <- sweep(data, 2, data_means, "-")
    data_scaled <- sweep(data_cent, 2, data_sds, "/")
    data_scaled <- data

    if (!is.null(seed)) {
        tf$keras$utils$set_random_seed(as.integer(seed))
    }
    n <- as.integer(dim(data)[1])
    p <- as.integer(dim(data)[2])
    aux_dim <- as.integer(dim(aux_data)[2])

    input_data <- layer_input(p)
    input_aux <- layer_input(aux_dim)
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
    input_decoder <- layer_concatenate(list(x_decoded_mean, input_aux))
    output_decoder <- input_decoder
    for (n_units in rev(hidden_units)) {
        dense_layer <- layer_dense(units = n_units, activation = activation)
        output_decoder <- output_decoder %>% dense_layer()
    }
    out_layer <- layer_dense(units = p)
    output_decoder <- output_decoder %>% out_layer()
    decoder <- keras_model(input_decoder, output_decoder)
    final_output <- layer_concatenate(list(output_decoder, z, z_mean_and_var))

    vae <- keras_model(list(input_data, input_aux), final_output)
    vae_loss <- function(x, res) {
        x_mean <- res[, 1:p]
        z_sample <- res[, (1 + p):(p + latent_dim)]
        z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
        z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
        log_px_z <- error_log_pdf(x, x_mean, tf$constant(error_dist_sigma, "float32"))
        log_qz_x <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
        log_pz <- source_log_pdf(z_sample, 0L, 1L)
        
        return(-tf$reduce_mean(log_px_z + log_pz - log_qz_x, -1L))
    }
    if (is.null(optimizer)) {
        optimizer <- tf$keras$optimizers$Adam(learning_rate = tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2))
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
        log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
        log_pz_u <- source_log_pdf(z_sample, 0, 1)
        return(-tf$reduce_mean((log_pz_u - log_qz_xu), -1L))
    })

    vae %>% compile(
        optimizer = optimizer,
        loss = vae_loss,
        metrics = list(metric_reconst_accuracy, metric_kl_vae)
    )
    if (!is.null(seed)) {
        tf$keras$utils$set_random_seed(as.integer(seed))
    }
    hist <- vae %>% fit(list(data_scaled, aux_data), data_scaled,
        shuffle = TRUE,
        validation_split = validation_split, batch_size = batch_size,
        epochs = epochs
    )


    IC_estimates <- predict(encoder, list(data_scaled, aux_data))
    IC_log_vars <- predict(z_log_var_model, list(data_scaled, aux_data))
    IC_means <- colMeans(IC_estimates)
    IC_sds <- apply(IC_estimates, 2, sd)
    IC_estimates_cent <- sweep(IC_estimates, 2, IC_means, "-")
    IC_estimates_scaled <- sweep(IC_estimates_cent, 2, IC_sds, "/")

    VAE_object <- list(
        IC_unscaled = IC_estimates, IC = IC_estimates_scaled,
        data_dim = p, metrics = hist,
        sample_size = n, call_params = call_params,
        encoder = encoder, decoder = decoder, IC_means = IC_means,
        IC_sds = IC_sds, call = deparse(sys.call()),
        D = paste(deparse(substitute(data)))
    )

    class(VAE_object) <- "CVAE"
    return(VAE_object)
}
