predict_coords_to_IC_ar1 <- function(
    object, last_spatial_locations, last_time_points, last_elevations = NULL, new_spatial_locations, new_time_points,
    new_elevation = NULL, get_var = FALSE, get_ar_coefs = FALSE) {
    if (!("iVAEradial_st" %in% class(object))) {
        stop("Object must be class of iVAEradial_st")
    }
    new_spatial_locations <- rbind(as.matrix(last_spatial_locations), as.matrix(new_spatial_locations))
    new_time_points <- c(last_time_points, new_time_points)
    order_inds <- order(new_time_points, new_spatial_locations[, 1], new_spatial_locations[, 2])
    original_inds <- order(order_inds)
    if (!is.null(new_elevation)) new_elevation <- c(last_elevations, new_elevation)
    #ord_coords_time <- cbind(new_spatial_locations, new_time_points)
    #if (!is.null(new_elevation)) ord_coords_time <- cbind(ord_coords_time, new_elevation)
    #ord_coords_time <- ord_coords_time[order_inds, ]
    N <- dim(new_spatial_locations)[1]
    new_spatial_locations_ord <- new_spatial_locations[order_inds, ]
    new_time_points_ord <- new_time_points[order_inds]
    if (!is.null(new_elevation)) {
        new_elevation_ord <- new_elevation[order_inds]
    } else {
        new_elevation_ord <- NULL
    }
    n_s_new <- nrow(as.data.frame(new_spatial_locations_ord) %>% dplyr::distinct())  
    phi_all <- get_aux_data_radial(object, new_spatial_locations_ord, new_time_points_ord, new_elevation_ord)
    if (get_var) {
        preds <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        preds <- sweep(preds, 2, object$IC_sds^2, "/")
    } else {
        # Retrieve last latent components based on spatial locations and time
        orig_coords_time <- cbind(object$locations, object$time)
        if (!is.null(object$elevation)) orig_coords_time <- cbind(orig_coords_time, object$elevation)
        last_ICs_inds <- which(object$locations[, 1] %in% new_spatial_locations[, 1] & 
                            object$locations[, 2] %in% new_spatial_locations[, 2] & 
                            object$time == max(object$time))
        last_IC_coords_time <- orig_coords_time[last_ICs_inds, ]
        last_ICs <- object$IC_unscaled[last_ICs_inds, ]
        last_IC_ord_inds <- order(last_IC_coords_time[, 3], last_IC_coords_time[, 1], last_IC_coords_time[, 2])
        last_ICs_ord <- last_ICs[last_IC_ord_inds, ]
        
        # Get AR(1) coefficients for the new locations and time points
        #phi_all_ord <- phi_all[order_inds, ]
        ar1_coeffs <- as.matrix(object$prior_ar1_model(phi_all))
        if (get_ar_coefs) {
            coefs <- ar1_coeffs[original_inds, ]
            coefs <- coefs[-(1:n_s_new), ]
            return(coefs)
        }

        prior_means <- as.matrix(object$prior_mean_model(phi_all))

        # Predict latent components using the AR(1) process
        preds <- matrix(0, nrow = (N), ncol = ncol(last_ICs_ord))
        
        unique_new_time_points <- sort(unique(new_time_points_ord))
        for (t in (seq_along(unique_new_time_points))) {
            if (t == 1) {
                preds[1:n_s_new, ] <- last_ICs_ord  # Initial prediction for the time step
            } else {
                start_ind <- (t - 1) * n_s_new + 1
                end_ind <- t * n_s_new
                prev_start_ind <- (t - 2) * n_s_new + 1
                prev_end_ind <- (t - 1) * n_s_new
                preds[start_ind:end_ind, ] <- prior_means[start_ind:end_ind, ] + ar1_coeffs[start_ind:end_ind, ] * (preds[prev_start_ind:prev_end_ind, ] - prior_means[prev_start_ind:prev_end_ind, ])
            }
        }
        preds <- preds[original_inds, ]
        preds <- preds[-(1:n_s_new), ]
        preds <- sweep(preds, 2, object$IC_means, "-")
        preds <- sweep(preds, 2, object$IC_sds, "/")
    }
    return(list(preds = preds, coords_time = cbind(new_spatial_locations[-(1:n_s_new), ], new_time_points[-(1:n_s_new)], new_elevation[-(1:n_s_new)])))
}


predict_coords_to_IC_ar1_train_data <- function(object, get_var = FALSE, get_ar_coefs = FALSE) {
    if (!("iVAEradial_st" %in% class(object))) {
        stop("Object must be class of iVAEradial_st")
    }
    N <- nrow(object$locations)
    n_t <- length(unique(object$time))
    n_s <- N/n_t
    time_ord <- order(object$time)
    orig_ord <- order(time_ord)
    phi_all <- get_aux_data_radial(object, object$locations, object$time, object$elevation)
    if (get_var) {
        preds <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        preds <- sweep(preds, 2, object$IC_sds^2, "/")
    } else {
        # Get AR(1) coefficients for the new locations and time points
        ar1_coeffs <- as.matrix(object$prior_ar1_model(phi_all))[time_ord, ]

        if (get_ar_coefs) return (ar1_coeffs[orig_ord, ])

        prior_means <- as.matrix(object$prior_mean_model(phi_all))[time_ord, ]
        prev_prior_means <- rbind(matrix(0, ncol = object$latent_dim, nrow = n_s), prior_means[-((N - n_s + 1): N), ])
        prev_ICs <- rbind(matrix(0, ncol = object$latent_dim, nrow = n_s), object$IC_unscaled[time_ord, ][-((N - n_s + 1): N), ])

        # Predict latent components using the AR(1) process
        preds <- prior_means + ar1_coeffs * (prev_ICs - prev_prior_means)
        preds <- sweep(preds, 2, object$IC_means, "-")
        preds <- sweep(preds, 2, object$IC_sds, "/")
        preds <- preds[orig_ord, ]
        #preds <- rbind(rep(0, object$latent_dim), preds[1:(nrow(preds) - 1), ])
    }
    return(preds)
}
