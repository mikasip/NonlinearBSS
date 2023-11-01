#' Time Contrastive Learning
#' @description Constructs and fits a model for time contrastive learning.
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param labels A vector of length N containing the labels
#' for the observations.
#' @param n_hidden_layers A number of hidden layers in TCL model.
#' @param n_hidden_units A number of hidden units in each hidden layer.
#' @param lr_start A starting learning rate.
#' @param lr_end A learning rate after polynomial decay.
#' @param seed Seed for the tensorflow model.
#' @param ... Further parameters for keras::fit function.
#' @return An object of class TCL.
#' @details The method constructs and fits a model for time contrastive learning
#' based on the given parameters. TCL assigns half of the labels incorrectly to
#' observations, and trains a deep neural network model to predict if the label
#' is correct or incorrect. The model has a bottleneck layer at the end, which
#' has P hidden units. The idea is, that in order to classify the data
#' correctly, it has to learn the true independent components. For more
#' details about TCL, see \insertCite{HyvarinenMorioka2016}{NonlinearBSS}.
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' @examples
#' p <- 10
#' n_segments <- 100
#' n_per_segment <- 50
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' labels <- numeric(n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (seg in 1:n_segments) {
#'     start_ind <- (seg - 1) * n_per_segment + 1
#'     end_ind <- seg * n_per_segment
#'     labels[start_ind:end_ind] <- seg
#'     for (i in 1:p) {
#'         latent_data[start_ind:end_ind, i] <- rnorm(
#'             n_per_segment,
#'             0, runif(1, 0.1, 5)
#'         )
#'     }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' 
#' # For better performance, increase the number of epochs.
#' res <- TCL(mixed_data, labels - 1,
#'     n_hidden_layers = 1,
#'     n_hidden_units = 32, batch_size = 64, epochs = 10
#' )
#' cormat <- cor(res$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @export
TCL <- function(
    data, labels, n_hidden_layers = 1,
    n_hidden_units = 32, lr_start = 0.01, lr_end = 0.001, seed = NULL, ...) {
    if (!is.null(seed)) {
        tf$keras$utils$set_random_seed(as.integer(seed))
    }

    n <- dim(data)[1]
    p <- dim(data)[2]
    data_means <- colMeans(data)
    data_sds <- apply(data, 2, sd)
    data_cent <- sweep(data, 2, data_means, "-")
    data_scaled <- sweep(data_cent, 2, data_sds, "/")
    x <- data_scaled

    y <- labels

    n_segments <- length(unique(labels))
    suffle_indexes <- sample(1:n)
    x_suffled <- x[suffle_indexes, ]
    y_suffled <- y[suffle_indexes]
    x_train <- x_suffled
    y_train <- y_suffled

    input <- layer_input(p)
    ICA_layer <- layer_dense(
        units = p, name = "ICA_Layer",
        activation = "linear"
    )

    submodel <- input
    for (i in 1:n_hidden_layers) {
        submodel <- submodel %>%
            layer_dense(
                units = n_hidden_units, activation = "leaky_relu"
            )
    }
    submodel_ICA <- submodel %>% ICA_layer()
    submodel_final <- submodel_ICA %>% absolute_activation()

    output <- submodel_final %>% layer_dense(n_segments,
        activation = "softmax"
    )

    model <- keras_model(inputs = input, outputs = output)
    ICA_model <- keras_model(inputs = input, outputs = ICA_layer$output)

    loss_fn <- loss_sparse_categorical_crossentropy()

    model %>% compile(
        optimizer = tf$keras$optimizers$legacy$Adam(
            learning_rate = tf$keras$optimizers$schedules$PolynomialDecay(
                lr_start, 10000, lr_end, 2
            )
        ),
        loss = loss_fn,
        metrics = "accuracy"
    )

    model %>% fit(x_train, y_train, shuffle = TRUE, ...)

    ICA_estimates <- predict(ICA_model, x)

    return(list(IC = ICA_estimates, mean = 0, whitening_matrix = 0))
}
