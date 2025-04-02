#' iVAEar Segmentation
#'
#' Fits an identifiable Variational Autoencoder with an autoregressive prior to segmented data. 
#' This function is designed for spatio-temporal data, where the data is divided into 
#' segments based on their spatio-temporal locations. The segmentation is used as auxiliary data.
#'
#' @param data A matrix of observed data (n x p) where each row is an observation and each column is a feature.
#' @param locations A matrix of spatio-temporal locations corresponding to each observation in \code{data}.
#' @param segment_sizes A vector providing sizes for segments.
#' The dimension should match the spatial dimenstion.
#' @param joint_segment_inds A vector indicating which segments
#' are considered jointly. See more in details.
#' @param latent_dim A latent dimension for iVAE.
#' @param n_s Number of unique spatial locations in the data.
#' @param ar_order An autoregressive order used in iVAEar.
#' @param epochs Integer specifying the number of training epochs for the iVAE model.
#' @param batch_size Integer specifying the batch size for training the iVAE model.
#' @param ... Additional arguments passed to the underlying \code{\link{iVAEar}} function.
#'
#' @return
#' An object of class iVAEspatial, inherits from class iVAE.
#' Additionally, the object has a property
#' \code{spatial_dim} which gives the dimension of the given locations.
#' For more details, see \code{\link{iVAE}}.
#' @details
#' The method creates the auxiliary data as spatio-temporal segments based on
#' the given input parameters.
#' The vector \code{segment_sizes} defines the size of the segments for
#' each spatial/temporal dimension separately. The segmentation is then created
#' based on the \code{joint_segment_inds}, which defines dimensions are
#' considered jointly. For example for spatio-temporal data with two spatial
#' dimensions, \code{joint_segment_inds = c(1, 1, 2)}
#' defines that the spatial dimensions are
#' considered jointly, and the temporal dimension is considered alone. 
#' This means that the auxiliary variable is vector
#' giving the two dimensional spatial segment and one dimensional temporal
#' segment in which the observation belongs in. All dimensions are considered jointly as
#' default.
#'
#' After the segmentation, the method calls the function \code{iVAEar}
#' using the created auxiliary variables.
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
#' resiVAE <- iVAEar_segmentation(
#'   data = obs_data, 
#'   locations = coords_time, 
#'   segment_sizes = c(0.1, 0.1, 5), 
#'   joint_segment_inds = c(1, 1, 2),
#'   latent_dim = p, 
#'   n_s = n_spat,
#'   epochs = 1,
#'   batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @seealso \code{\link{iVAEar}}
#' @author Mika Sipilä
#' @export
iVAEar_segmentation <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim, n_s,
    ar_order = 1, epochs, batch_size, ...) {
    n <- dim(data)[1]
    order_inds <- order(locations[, 3], locations[, 1], locations[, 2])
    original_order <- order(order_inds)
    data_ord <- data[order_inds, ]
    aux_data <- form_aux_data_spatial(locations, segment_sizes, joint_segment_inds)
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
        ar_order = ar_order, batch_size = batch_size, epochs = epochs, ...)
    class(resVAE) <- c("iVAEar1_spatial", class(resVAE))
    resVAE$spatial_dim <- dim(locations)[2]
    return(resVAE)
}
