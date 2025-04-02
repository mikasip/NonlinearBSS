#' Radial Basis Function Based Spatio Temporal Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder (iVAE) which uses
#' spatial and temporal radial basis functions as auxiliary data.
#' @importFrom rdist cdist
#' @importFrom Rdpack reprompt
#' @inheritDotParams iVAE -aux_data
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param spatial_locations A matrix containing spatial locations.
#' @param time_points A vector containing the time points.
#' @param latent_dim A latent dimension for iVAE.
#' @param elevation An optional elevation vector for the data.
#' @param spatial_dim The spatial dimension. Default is 2.
#' @param spatial_basis The spatial resolution levels to form the spatial
#' radial basis functions. Default is \code{c(2,9)}
#' @param temporal_basis The temporal resolution levels to form the temporal
#' radial basis functions. Default is \code{c(9, 17, 37)}
#' @param elevation_basis An optional vector of elevation resolution levels.
#' Must be given if \code{elevation} is provided for the data
#' @param seasonal_period An optional parameter giving the length of the
#' seasonal period.
#' @param spatial_kernel A kernel function to be used to form the spatial radial basis
#' functions. Either \code{"gaussian"} (default) or \code{"wendland"}.
#' @param epochs A number of epochs to train the model.
#' @param batch_size A batch size.
#' @return
#' An object of class iVAEradial_st, inherits from class iVAE.
#' Additionally, the object has the following properties:
#' \item{spatial_basis}{Same as the function input \code{spatial_basis}.}
#' \item{temporal_basis}{Same as the function input \code{temporal_basis}.}
#' \item{elevation_basis}{Same as the function input \code{elevation_basis}.}
#' \item{spatial_kernel}{Same as the function input \code{spatial_kernel}.}
#' \item{min_time_point}{Minimum time point.}
#' \item{max_time_point}{Maximum time point.}
#' \item{min_elevation}{Minimum elevation. \code{NULL}, if elevation is not
#' provided.}
#' \item{max_elevation}{Maximum elevation. \code{NULL}, if elevation is not
#' provided.}
#' \item{spatial_dim}{The spatial dimension.}
#' For more details, see \code{\link{iVAE}}.
#' @details
#' The method creates the auxiliary data as radial basis functions based on
#' the given input parameters.
#' The vectors \code{spatial_basis}, \code{temporal_basis} and
#' \code{elevation_basis} define the resolution levels
#' used to create the radial basis functions. If \code{seasonal_period}
#' is provided, the method uses the seasonal time index instead of the
#' absolute time index to form the temporal basis functions. Providing
#' \code{seasonal_perid} can be useful for e.g. forecasting purposes.
#'
#' After the forming the radial basis functions, the method
#' calls the function \code{iVAE} using the created auxiliary variables.
#'
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' @seealso
#' \code{\link{iVAE}}
#' \code{\link{generate_nonstationary_spatio_temporal_data_by_segments}}
#' @examples
#' p <- 3
#' n_time <- 100
#' n_spat <- 50
#' coords_time <- cbind(
#'     rep(runif(n_spat), n_time),
#'     rep(runif(n_spat), n_time), rep(1:n_time, each = n_spat)
#' )
#' data_obj <- generate_nonstationary_spatio_temporal_data_by_segments(
#'     n_time,
#'     n_spat, p, 5, 10, coords_time
#' )
#' latent_data <- data_obj$data
#' # Generate artificial observed data by applying a nonlinear mixture
#' obs_data <- mix_data(latent_data, 2)
#' cor(obs_data, latent_data)
#'
#' # For better peformance, increase the number of epochs.
#' resiVAE <- iVAE_radial_spatio_temporal(obs_data, coords_time[, 1:2],
#'     coords_time[, 3], p,
#'     epochs = 10, batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @export
iVAE_radial_spatio_temporal <- function(data, spatial_locations, time_points, latent_dim, 
    elevation = NULL, spatial_dim = 2, spatial_basis = c(2, 9), 
    temporal_basis = c(9, 17, 37), elevation_basis = NULL, seasonal_period = NULL, 
    spatial_kernel = "gaussian", epochs, batch_size, ...) {
    
    aux_data_obj <- form_radial_aux_data(spatial_locations, time_points, elevation, spatial_dim, spatial_basis, temporal_basis, elevation_basis, seasonal_period, spatial_kernel)
    resVAE <- iVAE(data, aux_data_obj$aux_data, latent_dim, epochs = epochs, batch_size = batch_size, ...)
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
