#' iVAEar with Radial Basis Functions for Spatial-Temporal Data
#'
#' Fits an identifiable Variational Autoencoder with an autoregressive prior (iVAEar) to spatio-temporal data. 
#' The method constructs the auxiliary data based on spatial and temporal radial basis functions.
#'
#' @param data A matrix of observed data (n x p), where each row is an observation and each column is a feature.
#' @param spatial_locations A matrix of spatial locations corresponding to each observation in \code{data}.
#' @param time_points A vector of time points corresponding to each observation in \code{data}.
#' @param latent_dim The latent dimension of iVAE.
#' @param n_s The number of unique spatial locations in the data.
#' @param elevation Optional vector of elevations corresponding to each observation in \code{data}.
#' Default is \code{NULL}.
#' @param spatial_dim An integer specifying the number of spatial dimensions. Default is 2 (e.g., latitude and longitude).
#' @param spatial_basis A vector specifying the number of spatial radial basis functions at different resolution levels.
#' @param temporal_basis A vector specifying the number of temporal radial basis functions at different resolution levels.
#' @param ar_order An autoregressive order used in iVAEar.
#' @param elevation_basis Optional vector specifying the number of elevation radial basis functions at different resolution levels. 
#' Default is \code{NULL}. Has to be provided if \code{elevation} is provided.
#' @param seasonal_period Optional integer specifying a seasonal period, if applicable (e.g., 12 for monthly data with a 
#' yearly period). Default is \code{NULL}.
#' @param max_season An integer giving the maximum number of seasonal periods considered. 
#' Can be used in case the forecasting period have a season number which is not present in the training data.
#' @param week_component A boolean value indicating if daily changes within a week is considered. 
#' Can be used only if the data is observed daily in time.
#' @param spatial_kernel A string specifying the kernel to use for spatial data. 
#' The options are \code{"gaussian"} and \code{"wendland"} Default is \code{"gaussian"}.
#' @param epochs Integer specifying the number of training epochs for the iVAE model.
#' @param batch_size Integer specifying the batch size for training the iVAE model.
#' @param ... Additional arguments passed to the underlying \code{\link{iVAEar}} function.
#'
#' @details 
#' This function applies radial basis functions (RBF) to the spatial, temporal, and possibly elevation data to create 
#' auxiliary variables. These auxiliary variables are passed into the iVAE model to capture dependencies in the latent space 
#' across space, time, and elevation.
#' 
#' For more details of forming radial basis function based, see \insertCite{sipila2024modelling}{NonlinearBSS}
#'
#' @return 
#' A fitted iVAEar object of class \code{iVAEradial_st}, which inherits from class \code{\link{iVAEar}}.
#' In addition the object has the following fields:
#'   \item{min_coords, max_coords}{Minimum and maximum coordinates of the spatial locations.}
#'   \item{min_time_point, max_time_point}{Minimum and maximum time points.}
#'   \item{min_elevation, max_elevation}{Minimum and maximum elevation values, if \code{elevation} is provided.}
#'   \item{spatial_basis}{The radial basis function configuration for spatial data.}
#'   \item{temporal_basis}{The radial basis function configuration for temporal data.}
#'   \item{elevation_basis}{The radial basis function configuration for elevation data, if applicable.}
#'   \item{spatial_kernel}{The kernel used for spatial data.}
#'   \item{seasonal_period}{The seasonal period if provided.}
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
#'
#' @seealso 
#' \code{\link{iVAEar}}
#' \code{\link{iVAEar_segmentation}}
#' @export
iVAEar_radial <- function(data, spatial_locations, time_points, latent_dim, n_s,
    elevation = NULL, spatial_dim = 2, spatial_basis = c(2, 9),
    temporal_basis = c(9, 17, 37), ar_order = 1, elevation_basis = NULL, 
    seasonal_period = NULL, max_season = NULL,
    week_component = FALSE, spatial_kernel = "gaussian", epochs, batch_size, ...) {
    n <- dim(data)[1]
    order_inds <- order(time_points, spatial_locations[, 1], spatial_locations[, 2])
    original_order <- order(order_inds)
    data_ord <- data[order_inds, ]
    aux_data_obj <- form_radial_aux_data(spatial_locations, time_points, elevation, spatial_dim, spatial_basis, temporal_basis, elevation_basis, seasonal_period, max_season, spatial_kernel, week_component)
    aux_data <- aux_data_obj$aux_data
    aux_data_ord <- aux_data[order_inds, ]
    data_prev_list <- list()
    aux_prev_list <- list()
    prev_data <- data_ord
    prev_data_aux <- aux_data_ord
    for (i in 1:ar_order) {
        data_prev_ord_i <- rbind(prev_data[1:n_s, ], prev_data[1:(n - n_s), ])
        prev_aux_data_ord_i <- rbind(prev_data_aux[1:n_s, ], prev_data_aux[1:(n - n_s), ])
        data_prev_i <- data_prev_ord_i[original_order, ]
        aux_prev_i <- prev_aux_data_ord_i[original_order, ]
        data_prev_list[[i]] <- data_prev_i
        aux_prev_list[[i]] <- aux_prev_i
        prev_data <- data_prev_ord_i
        prev_data_aux <- prev_aux_data_ord_i
    }
 
    resVAE <- iVAEar(data, aux_data, latent_dim, data_prev_list, aux_prev_list,
        ar_order = ar_order, epochs = epochs, batch_size = batch_size, ...)
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
    resVAE$locations <- spatial_locations
    resVAE$time <- time_points
    resVAE$elevation <- elevation
    resVAE$week_component <- week_component
    return(resVAE)
}
