#' Radial Basis Function Based Spatio Temporal Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder (iVAE) which uses
#' spatial and temporal radial basis functions as auxiliary data.
#' @import tensorflow
#' @import keras
#' @import rdist
#' @importFrom Rdpack reprompt
#' @inheritDotParams iVAE -aux_data
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param spatial_locations A matrix containing spatial locations.
#' @param time_points A vector containing the time points.
#' @param latent_dim A latent dimension for iVAE.
#' @param elevation An optional elevation vector for the data.
#' @param test_inds An optional vector of the indices of the rows used as
#' test data.
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
#' resiVAE <- iVAE_radial_spatio_temporal(obs_data, coords_time[1:2, ],
#'     coords_time[, 3], p,
#'     epochs = 10, batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @export
iVAE_radial_spatio_temporal <- function(data, spatial_locations, time_points, latent_dim, elevation = NULL, test_inds = NULL, spatial_dim = 2, spatial_basis = c(2, 9), temporal_basis = c(9, 17, 37), elevation_basis = NULL, seasonal_period = NULL, spatial_kernel = "gaussian", epochs, batch_size, ...) {
    spatial_kernel <- match.arg(spatial_kernel, c("gaussian", "wendland"))
    N <- dim(data)[1]
    min_coords <- apply(spatial_locations, 2, min)
    locations_new <- sweep(spatial_locations, 2, min_coords, "-")
    max_coords <- apply(locations_new, 2, max)
    locations_new <- sweep(locations_new, 2, max_coords, "/")
    knots_1d <- sapply(spatial_basis, FUN = function(i) seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i))
    phi_all <- matrix(0, ncol = 0, nrow = N)
    for (i in seq_along(spatial_basis)) {
        theta <- 1 / spatial_basis[i] * 2.5
        knot_list <- replicate(spatial_dim, knots_1d[[i]], simplify = FALSE)
        knots <- as.matrix(expand.grid(knot_list))
        phi <- cdist(locations_new, knots) / theta
        dist_leq_1 <- phi[which(phi <= 1)]
        dist_g_1_ind <- which(phi > 1)
        if (spatial_kernel == "gaussian") {
            phi <- gaussian_kernel(phi)
        } else {
            phi[which(phi <= 1)] <- wendland_kernel(dist_leq_1)
            phi[dist_g_1_ind] <- 0
        }
        phi_all <- cbind(phi_all, phi)
    }
    if (!is.null(seasonal_period)) {
        seasons <- floor(time_points / seasonal_period)
        seasons_model_matrix <- model.matrix(~ 0 + as.factor(seasons))
        phi_all <- cbind(phi_all, seasons_model_matrix)
        time_points <- time_points %% seasonal_period + 1
    }
    min_time_point <- min(time_points)
    max_time_point <- max(time_points)
    for (i in seq_along(temporal_basis)) {
        temp_knots <- c(seq(min_time_point, max_time_point, length.out = temporal_basis[i] + 2))
        temp_knots <- temp_knots[2:(length(temp_knots) - 1)]
        temp_dists <- cdist(time_points, temp_knots)
        kappa <- abs(temp_knots[1] - temp_knots[2])
        phi <- exp(-0.5 * (temp_dists)^2 / kappa^2)
        phi_all <- cbind(phi_all, phi)
    }
    min_elevation <- NULL
    max_elevation <- NULL
    if (!is.null(elevation)) {
        min_elevation <- min(elevation)
        max_elevation <- max(elevation)
    }
    if (!is.null(elevation) & !is.null(elevation_basis)) {
        for (i in seq_along(elevation_basis)) {
            elevation_knots <- c(seq(min_elevation, max_elevation, length.out = elevation_basis[i] + 2))
            elevation_knots <- elevation_knots[2:(length(elevation_knots) - 1)]
            elevation_dists <- cdist(elevation, elevation_knots)
            kappa <- abs(elevation_knots[1] - elevation_knots[2])
            phi <- exp(-0.5 * (elevation_dists)^2 / kappa^2)
            phi_all <- cbind(phi_all, phi)
        }
    }
    aux_data <- phi_all
    if (!is.null(test_inds)) {
        test_data <- data[test_inds, ]
        train_data <- data[-test_inds, ]
        test_aux_data <- data[test_inds, ]
        train_aux_data <- data[-test_inds, ]
    } else {
        test_data <- NULL
        train_data <- data
        test_aux_data <- NULL
        train_aux_data <- aux_data
    }
    resVAE <- iVAE(train_data, train_aux_data, latent_dim, test_data = test_data, test_data_aux = test_aux_data, epochs = epochs, batch_size = batch_size, get_prior_means = FALSE, ...)
    class(resVAE) <- c("iVAEradial_st", class(resVAE))
    resVAE$min_coords <- min_coords
    resVAE$max_coords <- max_coords
    if (!is.null(seasonal_period)) {
        resVAE$seasonal_period <- seasonal_period
        resVAE$max_season <- ifelse(is.null(seasonal_period), NULL, max(seasons))
    }
    resVAE$spatial_basis <- spatial_basis
    resVAE$temporal_basis <- temporal_basis
    resVAE$elevation_basis <- elevation_basis
    resVAE$spatial_kernel <- spatial_kernel
    resVAE$min_time_point <- min_time_point
    resVAE$max_time_point <- max_time_point
    resVAE$min_elevation <- min_elevation
    resVAE$max_elevation <- max_elevation
    resVAE$spatial_dim <- spatial_dim
    return(resVAE)
}
