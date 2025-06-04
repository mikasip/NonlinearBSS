#' Predict Latent Independent Components for New Spatial-Temporal Locations
#'
#' This function predicts the latent independent components (ICs) for new spatial-temporal locations 
#' based on the fitted autoregressive identifiable variational autoencoder (iVAEar).
#'
#' @param object An object of class `iVAEradial_st`.
#' @param last_spatial_locations A matrix of spatial coordinates corresponding to the last known time points.
#' @param last_time_points A vector of time points corresponding to `last_spatial_locations`.
#' @param last_elevations (Optional) A vector of elevation values corresponding to `last_spatial_locations`.
#' @param new_spatial_locations A matrix of new spatial coordinates where predictions are required.
#' @param new_time_points A vector of time points corresponding to `new_spatial_locations`.
#' @param new_elevation (Optional) A vector of elevation values corresponding to `new_spatial_locations`.
#' @param get_trend Logical; if `TRUE`, return the predicted trend functions of the ICs.
#' @param get_var Logical; if `TRUE`, return the predicted variance of the ICs.
#' @param get_ar_coefs Logical; if `TRUE`, return the predicted autoregressive (AR) coefficients of the ICs.
#' @param unscaled Logical; if `TRUE`, return unscaled predictions (raw ICs before standardization).
#'
#' @details This function utilizes an autoregressive iVAE model to predict 
#' latent independent components (ICs) at new spatial-temporal locations.
#' The method relies on a learned trend function \mjeqn{\mu(\mathbf{s}, t)}{ascii} 
#' and an autoregressive structure to propagate ICs over time.
#' 
#' Specifically, the model follows an autoregressive process of order \mjeqn{R}{ascii}:
#' 
#' \mjeqn{\mathbf{z}_t = \mu(\mathbf{s}, t) + \sum_{r=1}^{R} \gamma_r(\mathbf{s}, t) (\mathbf{z}_{t-r} - \mu(\mathbf{s}_t, t)) + \omega_t}{ascii}
#' 
#' where \mjeqn{\gamma_r(\mathbf{s}_t, t)}{ascii} are spatially varying autoregressive coefficients and 
#' \mjeqn{\omega_t}{ascii} represents stochastic noise.
#' 
#' Given new spatial and temporal coordinates, the function reconstructs ICs by 
#' sequentially applying the learned trend function and autoregressive model.
#'
#' @author Mika Sipilä
#' @return A list containing:
#'   - `preds`: A matrix of predicted ICs.
#'   - `coords_time`: A matrix of corresponding spatial and temporal coordinates (with optional elevation).
#'   - `trends`: A matrix of predicted mean of the ICs if `get_trend = TRUE`, otherwise `NULL`.
#'   - `vars`: A matrix of predicted variances of the ICs if `get_var = TRUE`, otherwise `NULL`.
#'   - `ar_coefs`: A matrix of predicted autoregressive coefficients of ICs if `get_ar_coefs = TRUE`, otherwise `NULL`.
#'
#' @export
predict_coords_to_IC_ar <- function(
    object, last_spatial_locations, last_time_points, last_elevations = NULL,
    new_spatial_locations, new_time_points, new_elevation = NULL, 
    get_trend = FALSE, get_var = FALSE, get_ar_coefs = FALSE, unscaled = FALSE) {
    if (!("iVAEradial_st" %in% class(object)) & !("iVAEar_segmentation" %in% class(object))) {
        stop("Object must be class of iVAEradial_st or iVAEar_segmentation")
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
    if ("iVAEar_radial" %in% class(object)) {
        phi_all <- get_aux_data_radial(object, new_spatial_locations_ord, new_time_points_ord, new_elevation_ord)
    } else {
        phi_all <- get_aux_data_spatial(object, cbind(new_spatial_locations_ord, new_time_points_ord))
    }
    if (get_var) {
        vars <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        if (!unscaled) {
            vars <- sweep(preds, 2, object$IC_sds^2, "/")
        }
    } else vars <- NULL
    # Retrieve last latent components based on spatial locations and time
    orig_coords_time <- cbind(object$locations, object$time)
    if (!is.null(object$elevation)) orig_coords_time <- cbind(orig_coords_time, object$elevation)
    last_ICs_inds <- which(object$locations[, 1] %in% new_spatial_locations[, 1] & 
                        object$locations[, 2] %in% new_spatial_locations[, 2] & 
                        object$time %in% unique(last_time_points))
    last_IC_coords_time <- orig_coords_time[last_ICs_inds, ]
    last_ICs <- object$IC_unscaled[last_ICs_inds, ]
    last_IC_ord_inds <- order(last_IC_coords_time[, 3], last_IC_coords_time[, 1], last_IC_coords_time[, 2])
    last_ICs_ord <- last_ICs[last_IC_ord_inds, ]
    
    # Get AR(1) coefficients for the new locations and time points
    #phi_all_ord <- phi_all[order_inds, ]
    ar_coeffs <- as.matrix(object$prior_ar_model(phi_all))
    if (get_ar_coefs) {
        coefs <- ar_coeffs[original_inds, ]
        coefs <- coefs[-(1:nrow(last_ICs_ord)), ]
    } else coefs <- NULL

    prior_means <- as.matrix(object$prior_mean_model(phi_all))
    if (get_trend) {
        if (!unscaled) {
            trend <- sweep(prior_means, 2, object$IC_means, "-")
            trend <- sweep(trend, 2, object$IC_sds, "/")
        }
    } else trend <- NULL

    # Predict latent components using the AR(1) process
    preds <- matrix(0, nrow = (N), ncol = ncol(last_ICs_ord))
        
    unique_new_time_points <- sort(unique(new_time_points_ord))
    preds[1:nrow(last_ICs_ord), ] <- last_ICs_ord 
    for (t in (seq_along(unique_new_time_points))) {
        if (t > object$ar_order) {
            start_ind <- (t - 1) * n_s_new + 1
            end_ind <- t * n_s_new
            pred <- prior_means[start_ind:end_ind, ]
            for (i in 1:object$ar_order) {
                ar_coef <- ar_coeffs[start_ind:end_ind, ((i - 1) * object$latent_dim + 1):(i * object$latent_dim)]
                pred_i <- preds[(start_ind - (i * n_s_new)):(end_ind - (i * n_s_new)), ]
                mean_i <- prior_means[(start_ind - (i * n_s_new)):(end_ind - (i * n_s_new)), ]
                pred <- pred + ar_coef * (pred_i - mean_i)
            }
            preds[start_ind:end_ind, ] <- pred
        }
    }
    preds <- preds[original_inds, ]
    preds <- preds[-(1:nrow(last_ICs_ord)), ]
    if (!unscaled) {
        preds <- sweep(preds, 2, object$IC_means, "-")
        preds <- sweep(preds, 2, object$IC_sds, "/")
    }
    return(list(preds = preds, coords_time = cbind(new_spatial_locations[-(1:n_s_new), ], 
            new_time_points[-(1:n_s_new)], new_elevation[-(1:n_s_new)]),
            trends = trend, vars = vars, ar_coefs = coefs))
}

#' Predict Latent Independent Components for Training Data
#'
#' This function predicts the latent independent components (ICs) for the training data locations 
#' using the fitted autoregressive identifiable variational autoencoder (iVAEar).
#'
#' @param object An object of class `iVAEradial_st`.
#' @param get_trend Logical; if `TRUE`, return the predicted trend functions of the ICs.
#' @param get_var Logical; if `TRUE`, return the predicted variance of the ICs instead of the IC predictions.
#' @param get_ar_coefs Logical; if `TRUE`, return the autoregressive (AR) coefficients instead of IC predictions.
#' @param unscaled Logical; if `TRUE`, return unscaled predictions (raw ICs before standardization).
#'
#' @inherit predict_coords_to_IC_ar details
#' @author Mika Sipilä
#' @return A list containing:
#'   - `preds`: A matrix of predicted ICs.
#'   - `trends`: A matrix of predicted mean of the ICs if `get_trend = TRUE`, otherwise `NULL`.
#'   - `vars`: A matrix of predicted variances of the ICs if `get_var = TRUE`, otherwise `NULL`.
#'   - `ar_coefs`: A matrix of predicted autoregressive coefficients of ICs if `get_ar_coefs = TRUE`, otherwise `NULL`.
#' 
#' @export
predict_coords_to_IC_ar_train_data <- function(object, get_var = FALSE, 
    get_ar_coefs = FALSE, unscaled = FALSE) {
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
        vars <- exp(as.matrix(object$prior_log_var_model(phi_all)))
        if (!unscaled) {
            preds <- sweep(vars, 2, object$IC_sds^2, "/")
        }
    } else vars <- NULL

    # Get AR(1) coefficients for the new locations and time points
    ar_coeffs <- as.matrix(object$prior_ar_model(phi_all))[time_ord, ]

    if (get_ar_coefs) return (ar_coeffs[orig_ord, ])

    prior_means <- as.matrix(object$prior_mean_model(phi_all))[time_ord, ]
    prev_prior_mean_list <- list()
    prev_ICs_list <- list()
    prev_ICs <- object$IC_unscaled[time_ord, ]
    prev_prior_means <- prior_means
    latent_dim <- object$call_params$latent_dim
    for (i in 1:object$ar_order) {
        prev_prior_means <- rbind(matrix(0, ncol = latent_dim, nrow = n_s), prev_prior_means[-((N - n_s + 1): N), ])
        prev_ICs <- rbind(matrix(0, ncol = latent_dim, nrow = n_s), prev_ICs[-((N - n_s + 1): N), ])
        prior_means <- prior_means + ar_coeffs[, ((i - 1) * latent_dim + 1):(i * latent_dim)] * (prev_ICs - prev_prior_means)
    }
    # Predict latent components using the AR(1) process
    preds <- prior_means
    if (!unscaled) {
        preds <- sweep(preds, 2, object$IC_means, "-")
        preds <- sweep(preds, 2, object$IC_sds, "/")
    }
    preds <- preds[orig_ord, ]
    #preds <- rbind(rep(0, object$latent_dim), preds[1:(nrow(preds) - 1), ])
    return(preds)
}


