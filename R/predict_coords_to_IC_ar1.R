predict_coords_to_IC_ar1 <- function(
    object, new_spatial_locations, new_time_points,
    new_elevation = NULL, get_var = FALSE) {
    if (!("iVAEradial_st" %in% class(object))) {
        stop("Object must be class of iVAEradial_st")
    }
    N <- dim(new_spatial_locations)[1]
    n_s_new <- nrow(as.data.frame(new_spatial_locations) %>% dplyr::distinct())
    phi_all <- get_aux_data_radial(object, new_spatial_locations, new_time_points, new_elevation)
    if (get_var) {
        preds <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        preds <- sweep(preds, 2, object$IC_sds^2, "/")
    } else {
        # Retrieve last latent components based on spatial locations and time
        last_ICs <- object$IC[object$locations[, 1] %in% new_spatial_locations[, 1] & 
                            object$locations[, 2] %in% new_spatial_locations[, 2] & 
                            object$time == max(object$time), ]
        # Get AR(1) coefficients for the new locations and time points
        ar1_coeffs <- as.matrix(object$prior_ar1_model(phi_all))
        
        # Predict latent components using the AR(1) process
        preds <- matrix(0, nrow = N, ncol = ncol(last_ICs))
        
        unique_new_time_points <- sort(unique(new_time_points))
        for (t in seq_along(unique_new_time_points)[-length(unique_new_time_points)]) {
            if (t == 1) {
                preds[1:n_s_new, ] <- ar1_coeffs[1:n_s_new, ] * last_ICs  # Initial prediction for the first time step
            } else {
                start_ind <- t * n_s_new + 1
                end_ind <- (t + 1) * n_s_new
                prev_start_ind <- (t - 1) * n_s_new + 1
                prev_end_ind <- t * n_s_new
                preds[start_ind:end_ind, ] <- ar1_coeffs[start_ind:end_ind, ] * preds[prev_start_ind:prev_end_ind, ]
            }
        }
    }
    return(preds)
}

predict_coords_to_IC_ar1_train_data <- function(object, get_var = FALSE) {
    if (!("iVAEradial_st" %in% class(object))) {
        stop("Object must be class of iVAEradial_st")
    }
    N <- nrow(object$locations)
    phi_all <- get_aux_data_radial(object, object$locations, object$time, object$elevation)
    if (get_var) {
        preds <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        preds <- sweep(preds, 2, object$IC_sds^2, "/")
    } else {
        # Get AR(1) coefficients for the new locations and time points
        ar1_coeffs <- as.matrix(object$prior_ar1_model(phi_all))

        # Predict latent components using the AR(1) process
        preds <- ar1_coeffs * (object$IC_unscaled)

        preds <- sweep(preds, 2, object$IC_means, "-")
        preds <- sweep(preds, 2, object$IC_sds, "/")
        preds <- rbind(rep(0, object$latent_dim), preds[1:(nrow(preds) - 1), ])
        preds[which(object$time == min(object$time)), ] <- rep(0, object$latent_dim)
    }
    return(preds)
}
