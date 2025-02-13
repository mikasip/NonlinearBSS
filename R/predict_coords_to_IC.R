#' Predict the Latent Components from Coordinates
#' @description Predicts the latent components from the coordinates using
#' an object of class \code{iVAE_radial_st}
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#'
#' @param object An object of class \code{iVAE_radial_st}
#' @param new_spatial_locations A matrix containing the new spatial locations.
#' @param new_time_points A vector containing the new temporal points.
#' @param new_elevation An optional vector containing elevation for new data.
#' @param get_var A boolean. If \code{TRUE}, the method returns the
#' spatio-temporal variance function estimate.
#' Otherwise returns the trend function estimate.
#' @param unscaled A boolean. If \code{TRUE}, the method returns 
#' the latent components or variance estimate as unscaled.
#' @return
#' A matrix containing the predicted variance or trend function for
#' the latent components.
#' @details
#' The method uses trained spatio-temporal identifiable variational autoencoder
#' to predict the latent components directly using new coordinates provided
#' in the function input.
#'
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' @seealso
#' \code{\link{iVAE_radial_spatio_temporal}}
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
#' # For better peformance, increase the number of epochs.
#' resiVAE_st <- iVAE_radial_spatio_temporal(obs_data, coords_time[, 1:2], coords_time[, 3], p,
#'     epochs = 10, batch_size = 64
#' )
#'
#' new_spatial_coords <- cbind(runif(10), runif(10))
#' new_time_points <- rep(1, 10)
#'
#' trend_estimate <- predict_coords_to_IC(
#'     resiVAE_st, new_spatial_coords,
#'     new_time_points
#' )
#' variance_estimate <- predict_coords_to_IC(resiVAE_st, new_spatial_coords,
#'     new_time_points,
#'     get_var = TRUE
#' )
#' @export
predict_coords_to_IC <- function(
    object, new_spatial_locations, new_time_points,
    new_elevation = NULL, get_var = FALSE, unscaled = FALSE) {
    if (!("iVAEradial_st" %in% class(object))) {
        stop("Object must be class of iVAEradial_st")
    }
    N <- dim(new_spatial_locations)[1]
    locations_new <- sweep(new_spatial_locations, 2, object$min_coords, "-")
    locations_new <- sweep(locations_new, 2, object$max_coords, "/")
    knots_1d <- sapply(object$spatial_basis, FUN = function(i) {
        seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i)
    })
    phi_all <- matrix(0, ncol = 0, nrow = N)
    for (i in seq_along(object$spatial_basis)) {
        theta <- 1 / object$spatial_basis[i] * 2.5
        knot_list <- replicate(object$spatial_dim, knots_1d[[i]], simplify = FALSE)
        knots <- as.matrix(expand.grid(knot_list))
        phi <- cdist(locations_new, knots) / theta
        dist_leq_1 <- phi[which(phi <= 1)]
        dist_g_1_ind <- which(phi > 1)
        if (object$spatial_kernel == "gaussian") {
            phi <- gaussian_kernel(phi)
        } else {
            phi[which(phi <= 1)] <- wendland_kernel(dist_leq_1)
            phi[dist_g_1_ind] <- 0
        }
        phi_all <- cbind(phi_all, phi)
    }
    if (object$week_component) {
        day_of_week <- new_time_points %% 7
        day_of_week_model_matrix <- model.matrix(~ 0 + as.factor(day_of_week))
        phi_all <- cbind(phi_all, day_of_week_model_matrix)
    }
    if (!is.null(object$seasonal_period)) {
        seasons <- c(
            0:object$max_season,
            floor(new_time_points / object$seasonal_period)
        )
        seasons[seasons > object$max_season] <- object$max_season
        seasons_model_matrix <- model.matrix(~ 0 + as.factor(seasons))
        phi_all <- cbind(
            phi_all,
            seasons_model_matrix[-c(1:(object$max_season + 1)), ]
        )
        new_time_points <- new_time_points %% object$seasonal_period + 1
    }
    for (i in seq_along(object$temporal_basis)) {
        temp_knots <- c(seq(object$min_time_point, object$max_time_point,
            length.out = object$temporal_basis[i] + 2
        ))
        temp_knots <- temp_knots[2:(length(temp_knots) - 1)]
        temp_dists <- cdist(new_time_points, temp_knots)
        kappa <- abs(temp_knots[1] - temp_knots[2])
        phi <- exp(-0.5 * (temp_dists)^2 / kappa^2)
        phi_all <- cbind(phi_all, phi)
    }
    if (!is.null(new_elevation)) {
        for (i in seq_along(object$elevation_basis)) {
            elevation_knots <- c(seq(object$min_elevation, object$max_elevation,
                length.out = object$elevation_basis[i] + 2
            ))
            elevation_knots <- elevation_knots[2:(length(elevation_knots) - 1)]
            elevation_dists <- cdist(new_elevation, elevation_knots)
            kappa <- abs(elevation_knots[1] - elevation_knots[2])
            phi <- exp(-0.5 * (elevation_dists)^2 / kappa^2)
            phi_all <- cbind(phi_all, phi)
        }
    }
    if (get_var) {
        preds <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        if (!unscaled) {
            preds <- sweep(preds, 2, object$IC_sds^2, "/")
        }
    } else {
        preds <- as.matrix(object$prior_mean_model(phi_all))
        if (!unscaled) {
            preds <- sweep(preds, 2, object$IC_means, "-")
            preds <- sweep(preds, 2, object$IC_sds, "/")
        }
    }
    return(preds)
}
