#' iVAEar1 with Radial Basis Functions for Spatial-Temporal Data
#'
#' Fits an identifiable Variational Autoencoder with an autoregressive prior (iVAEar1) to spatial-temporal data, 
#' incorporating radial basis functions for spatial and temporal covariates. This method allows the iVAE model 
#' to handle structured data with complex spatial, temporal, and elevation dependencies.
#'
#' @param data A matrix of observed data (n x p), where each row is an observation and each column is a feature.
#' @param spatial_locations A matrix of spatial locations corresponding to each observation in \code{data}. Each row 
#' represents spatial coordinates (e.g., latitude and longitude for spatial data).
#' @param time_points A vector of time points corresponding to each observation in \code{data}.
#' @param latent_dim An integer specifying the number of latent dimensions for the iVAE model.
#' @param n_s The number of previous observations used to form the autoregressive prior.
#' @param elevation Optional vector of elevation data corresponding to each observation in \code{data}.
#' Default is \code{NULL}.
#' @param test_inds An optional vector of indices for test data. If provided, the data at these indices will be held 
#' out for evaluation as a test set. Default is \code{NULL}.
#' @param spatial_dim An integer specifying the number of spatial dimensions. Default is 2 (e.g., latitude and longitude).
#' @param spatial_basis A vector specifying the number of radial basis functions to use for spatial data. 
#' Default is \code{c(2, 9)}.
#' @param temporal_basis A vector specifying the number of radial basis functions to use for temporal data. 
#' Default is \code{c(9, 17, 37)}.
#' @param elevation_basis Optional vector specifying the number of radial basis functions for elevation data. 
#' Default is \code{NULL}.
#' @param seasonal_period Optional integer specifying a seasonal period, if applicable (e.g., 12 for monthly data with a 
#' yearly period). Default is \code{NULL}.
#' @param spatial_kernel A string specifying the kernel to use for spatial data. Default is \code{"gaussian"}.
#' @param epochs Integer specifying the number of training epochs for the iVAE model.
#' @param batch_size Integer specifying the batch size for training the iVAE model.
#' @param ... Additional arguments passed to the underlying \code{\link{iVAEar1}} function, including other hyperparameters 
#' and model architecture specifications.
#'
#' @details 
#' This function applies radial basis functions (RBF) to the spatial, temporal, and possibly elevation data to create 
#' auxiliary variables. These auxiliary variables are passed into the iVAE model to capture dependencies in the latent space 
#' across space, time, and elevation.
#'
#' @return 
#' A fitted iVAEar1 object of class \code{"iVAEradial_st"}, which includes:
#' \itemize{
#'   \item \code{min_coords}, \code{max_coords}: Minimum and maximum coordinates of the spatial locations.
#'   \item \code{min_time_point}, \code{max_time_point}: Minimum and maximum time points.
#'   \item \code{min_elevation}, \code{max_elevation}: Minimum and maximum elevation values, if \code{elevation} is provided.
#'   \item \code{spatial_basis}: The radial basis function configuration for spatial data.
#'   \item \code{temporal_basis}: The radial basis function configuration for temporal data.
#'   \item \code{elevation_basis}: The radial basis function configuration for elevation data, if applicable.
#'   \item \code{spatial_kernel}: The kernel used for spatial data.
#'   \item \code{seasonal_period}: The seasonal period if provided.
#'   \item Other components from the \code{\link{iVAEar1}} model, such as encoder, decoder, and latent variable estimates.
#' }
#'
#' @examples
#' # Example usage with spatial-temporal data
#' data <- matrix(rnorm(1000), nrow = 100, ncol = 10)  # Simulated data
#' spatial_locations <- matrix(runif(200), nrow = 100, ncol = 2)  # Random 2D locations
#' time_points <- seq(1, 100)  # Sequential time points
#' latent_dim <- 3  # 3 latent dimensions
#' n_s <- 100
#'
#' result <- iVAEar1_radial(
#'   data = data, 
#'   spatial_locations = spatial_locations, 
#'   time_points = time_points, 
#'   latent_dim = latent_dim, 
#'   n_s = n_s, 
#'   epochs = 50, 
#'   batch_size = 32
#' )
#' print(result)
#'
#' @seealso \code{\link{iVAEar1}} for the core iVAE with an autoregressive prior model.
#' @export
iVAEar1_radial <- function(data, spatial_locations, time_points, latent_dim, n_s,
    elevation = NULL, test_inds = NULL, spatial_dim = 2, spatial_basis = c(2, 9), 
    temporal_basis = c(9, 17, 37), elevation_basis = NULL, seasonal_period = NULL, 
    spatial_kernel = "gaussian", epochs, batch_size, ...) {
    n <- dim(data)[1]
    aux_data_obj <- form_radial_aux_data(spatial_locations, time_points, elevation, test_inds, spatial_dim, spatial_basis, temporal_basis, elevation_basis, seasonal_period, spatial_kernel)
    if (!is.null(test_inds)) {
        test_data <- data[test_inds, ]
        train_data <- data[-test_inds, ]
        test_aux_data <- aux_data_obj$aux_data[test_inds, ]
        train_aux_data <- aux_data_obj$aux_data[-test_inds, ]
    } else {
        test_data <- NULL
        train_data <- data
        test_aux_data <- NULL
        train_aux_data <- aux_data_obj$aux_data
    }
    data_prev <- rbind(data[1:n_s, ], data[1:(n - n_s), ])
    resVAE <- iVAEar1(train_data, train_aux_data, latent_dim, data_prev = data_prev,
        test_data = test_data, test_data_aux = test_aux_data, epochs = epochs, 
        batch_size = batch_size, get_prior_means = FALSE, ...)
    class(resVAE) <- c("iVAEradial_st", class(resVAE))
    resVAE$min_coords <- aux_data_obj$min_coords
    resVAE$max_coords <- aux_data_obj$max_coords
    if (!is.null(seasonal_period)) {
        resVAE$seasonal_period <- seasonal_period
        resVAE$max_season <- ifelse(is.null(seasonal_period), NULL, max(aux_data_obj$seasons))
    }
    resVAE$spatial_basis <- spatial_basis
    resVAE$temporal_basis <- temporal_basis
    resVAE$elevation_basis <- elevation_basis
    resVAE$spatial_kernel <- aux_data_obj$spatial_kernel
    resVAE$min_time_point <- aux_data_obj$min_time_point
    resVAE$max_time_point <- aux_data_obj$max_time_point
    resVAE$min_elevation <- aux_data_obj$min_elevation
    resVAE$max_elevation <- aux_data_obj$max_elevation
    resVAE$spatial_dim <- spatial_dim
    return(resVAE)
}
