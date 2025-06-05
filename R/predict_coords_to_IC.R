#' Predict the Latent Components from Coordinates
#' @description Predicts the latent components from the coordinates using
#' an object of class \code{iVAE_radial_st}
#' @importFrom magrittr %>%
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
    if (!("iVAEradial_st" %in% class(object)) & !("iVAEar_segmentation" %in% class(object))) {
        stop("Object must be class of iVAEradial_st or iVAEar_segmentation")
    }
    N <- dim(new_spatial_locations)[1]
    if ("iVAEradial_st" %in% class(object)) {
        phi_all <- get_aux_data_radial(object, new_spatial_locations, new_time_points, new_elevation)
    } else {
        phi_all <- get_aux_data_spatial(object, cbind(new_spatial_locations, new_time_points))
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
