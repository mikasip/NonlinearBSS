#' iVAEar1 Segmentation
#'
#' Fits an identifiable Variational Autoencoder with an autoregressive prior (iVAEar1) to segmented data. 
#' This function is designed for spatial or temporal segmentation, where the data is divided into regions 
#' or time blocks, and auxiliary data is constructed from spatial or temporal information.
#'
#' @param data A matrix of observed data (n x p) where each row is an observation and each column is a feature.
#' @param locations A matrix of spatial or temporal locations corresponding to each observation in \code{data}.
#' Each row represents the coordinates (e.g., latitude and longitude for spatial data, or time indices for temporal data).
#' @param segment_sizes A vector of sizes of each segment in the data, indicating the number of observations in each segment.
#' @param joint_segment_inds A vector of indices (default: \code{rep(1, length(segment_sizes))}) specifying which 
#' segments should be treated as part of the same joint process. This allows combining segments.
#' @param latent_dim An integer specifying the number of latent dimensions for the iVAE model.
#' @param n_s The number of previous observations used to form the autoregressive prior. This corresponds to the 
#' number of time steps or observations used for conditioning in the latent space.
#' @param test_inds An optional vector of indices for test data. If provided, the data at these indices will be held 
#' out for evaluation as a test set. Default is \code{NULL}.
#' @param ... Additional arguments passed to the underlying \code{\link{iVAEar1}} function, including hyperparameters 
#' such as epochs, batch size, and model architecture specifications.
#'
#' @details 
#' The function constructs auxiliary data based on the spatial or temporal structure of the observations, 
#' specified by \code{locations} and \code{segment_sizes}. This auxiliary data is used as an input to the 
#' variational autoencoder, allowing the model to account for spatial or temporal dependencies. The function 
#' also supports test data, which can be specified via the \code{test_inds} argument.
#'
#' The autoregressive prior in the iVAE model is constructed by using \code{n_s} prior observations, allowing 
#' the model to capture time-dependent or location-dependent latent dynamics across the segments.
#'
#' @return 
#' A fitted iVAEar1 object of class \code{"iVAEar1_spatial"}, which includes:
#' \itemize{
#'   \item \code{spatial_dim}: The dimensionality of the \code{locations} data (e.g., 2 for spatial, 1 for temporal).
#'   \item \code{test_data}: The test data, if \code{test_inds} is provided.
#'   \item \code{test_aux_data}: The auxiliary data corresponding to the test set, if \code{test_inds} is provided.
#'   \item Other components from the \code{\link{iVAEar1}} model, such as encoder, decoder, and latent variable estimates.
#' }
#'
#' @examples
#' # Example usage with spatial data
#' data <- matrix(rnorm(1000), nrow = 100, ncol = 10)  # Simulated data
#' locations <- matrix(runif(200), nrow = 100, ncol = 2)  # Random 2D locations
#' segment_sizes <- rep(10, 10)  # 10 segments of size 10
#' latent_dim <- 3  # 3 latent dimensions
#' n_s <- 5  # Use 5 previous observations for the autoregressive prior
#'
#' result <- iVAEar1_segmentation(
#'   data = data, 
#'   locations = locations, 
#'   segment_sizes = segment_sizes, 
#'   latent_dim = latent_dim, 
#'   n_s = n_s
#' )
#' print(result)
#'
#' @seealso \code{\link{iVAEar1}} for the core iVAE with an autoregressive prior model.
#' @export
iVAEar1_segmentation <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim, n_s,
    test_inds = NULL, ...) {
    n <- dim(data)[1]
    aux_data <- form_aux_data_spatial(locations, segment_sizes, joint_segment_inds)
    test_data <- NULL
    test_aux_data <- NULL
    if (!is.null(test_inds)) {
        test_data <- data[test_inds, ]
        test_aux_data <- aux_data[test_inds, ]
    }
    data_prev <- rbind(data[1:n_s, ], data[1:(n - n_s), ])
    resVAE <- iVAEar1(data, aux_data, latent_dim,
        data_prev = data_prev, test_data = test_data,
        test_data_aux = test_aux_data, ...
    )
    class(resVAE) <- c("iVAEar1_spatial", class(resVAE))
    resVAE$spatial_dim <- dim(locations)[2]
    resVAE$test_data <- test_data
    resVAE$test_aux_data <- test_aux_data
    return(resVAE)
}