#' iVAEar Segmentation
#'
#' @description \loadmathjax Fits an identifiable Variational Autoencoder with an autoregressive prior to segmented data. 
#' This function is designed for spatio-temporal data, where the data is divided into 
#' segments based on their spatio-temporal locations. The segmentation is used as auxiliary data.
#'
#' @param data A matrix of observed data (n x p) where each row is an observation and each column is a feature.
#' @param locations A matrix of spatio-temporal locations corresponding to each observation in \code{data}. Not provided for STFDF object.
#' @param segment_sizes A vector providing sizes for segments.
#' The dimension should match the spatial dimenstion.
#' @param joint_segment_inds A vector indicating which segments
#' are considered jointly. See more in details.
#' @param var_names For STFDF objects only. A character vector specifying which variables to use from the STFDF data. 
#'   If NULL, all variables will be used.
#' @param elevation_var For STFDF objects only. The name of the variable in the STFDF object that contains elevation data.
#'   If NULL, no elevation data will be used.
#' @param latent_dim A latent dimension for iVAE.
#' @param n_s Number of unique spatial locations in the data. Not provided for STDFD object.
#' @param time_dim The index of the time dimension in the \code{locations} matrix.
#' Has to be provided if \code{seasonal_period} or \code{week_component} is not NULL.
#' @param seasonal_period The length of the seasonal period in the time dimension.
#' If provided, the temporal segmentation is done based on day of the year.
#' @param max_season An integer giving the maximum number of seasonal periods considered. 
#' Can be used in case the forecasting period have a season number which is not present in the training data.
#' @param week_component A boolean value indicating if daily changes within a week is considered. 
#' Can be used only if the data is observed daily in time.
#' @param ar_order An autoregressive order used in iVAEar.
#' @param epochs Integer specifying the number of training epochs for the iVAE model.
#' @param batch_size Integer specifying the batch size for training the iVAE model.
#' @param ... Additional arguments passed to the underlying \code{\link{iVAEar}} function.
#'
#' @return
#' An object of class iVAEspatial, inherits from class iVAE.
#' Additionally, the object has a property
#' \item{spatial_dim}{The dimension of the spatial locations}
#' \item{original_stfdf}{For STFDF objects, the original input STFDF object is stored here.}
#' For more details, see \code{\link{iVAE}}.
#'
#' @details
#' This function implements the segmentation-based identifiable VAE with autoregressive latent components.
#' 
#' In segmentation-based iVAE, a spatio-temporal segmentation is used as an auxiliary variable. The 
#' spatio-temporal domain \mjeqn{\mathcal{S} \times \mathcal{T}}{ascii} is divided into \mjeqn{m}{ascii} non-intersecting 
#' segments \mjeqn{\mathcal{K}_i \in \mathcal{S} \times \mathcal{T}}{ascii} such that 
#' \mjeqn{\mathcal{K}_i \cap \mathcal{K}_j = \emptyset}{ascii} for all \mjeqn{i \neq j}{ascii}, 
#' \mjeqn{i,j = 1,\dots,m}{ascii}, and \mjeqn{\cup_{i=1}^m \mathcal{K}_i = \mathcal{S} \times \mathcal{T}}{ascii}.
#' 
#' Using an indicator function \mjeqn{\mathbbm{1}}{ascii}, the auxiliary variable for the observation 
#' \mjeqn{\mathbf{x}(\mathbf{s}, t)}{ascii} can be written as:
#' \mjeqn{\mathbf{u}(\mathbf{s}, t) = (\mathbbm{1}((\mathbf{s}, t) \in \mathcal{K}_1), \dots, \mathbbm{1}((\mathbf{s}, t) \in \mathcal{K}_m))^\top}{ascii}
#' 
#' where \mjeqn{\mathbbm{1}((\mathbf{s}, t) \in \mathcal{K}_i) = 1}{ascii} if the location 
#' \mjeqn{(\mathbf{s}, t)}{ascii} is within the segment \mjeqn{\mathcal{K}_i}{ascii}, and 0 otherwise. This results 
#' in an \mjeqn{m}{ascii}-dimensional standard basis vector, where the value 1 indicates the 
#' spatio-temporal segment to which the observation belongs.
#' 
#' To reduce dimensionality when the spatio-temporal domain is large and small segments are used, 
#' the spatial and temporal segmentations can be considered separately:
#' 
#' - The auxiliary data is composed of \mjeqn{m_S}{ascii} spatial segments \mjeqn{\mathcal{S}_i \in \mathcal{S}}{ascii} 
#'   and \mjeqn{m_T}{ascii} temporal segments \mjeqn{\mathcal{T}_i \in \mathcal{T}}{ascii}
#' - \mjeqn{\mathcal{S}_i \cap \mathcal{S}_j = \emptyset}{ascii} for all \mjeqn{i \neq j}{ascii}, \mjeqn{i,j = 1,\dots,m_S}{ascii}
#' - \mjeqn{\cup_{i=1}^{m_S} \mathcal{S}_i = \mathcal{S}}{ascii}
#' - \mjeqn{\mathcal{T}_i \cap \mathcal{T}_j = \emptyset}{ascii} for all \mjeqn{i \neq j}{ascii}, \mjeqn{i,j = 1,\dots,m_T}{ascii}
#' - \mjeqn{\cup_{i=1}^{m_T} \mathcal{T}_i = \mathcal{T}}{ascii}
#' 
#' Then, the auxiliary variable for the observation \mjeqn{\mathbf{x}(\mathbf{s}, t)}{ascii} becomes:
#' \mjeqn{\mathbf{u}(\mathbf{s}, t) = (\mathbbm{1}(\mathbf{s} \in \mathcal{S}_1), \dots, \mathbbm{1}(\mathbf{s} \in \mathcal{S}_{m_S}), \mathbbm{1}(t \in \mathcal{T}_1), \dots, \mathbbm{1}(t \in \mathcal{T}_{m_T}))^\top}{ascii}
#' 
#' This auxiliary variable is \mjeqn{(m_S + m_T)}{ascii}-dimensional and has two nonzero entries for each observation.
#' The dimension can be reduced even further by considering the x-axis and y-axis of the spatial domain separately.
#' 
#' The function implements three variants of segmentation-based iVAE:
#' 
#' - iVAEs1: All dimensions (x-axis, y-axis, time) segmented separately
#' - iVAEs2: Space and time segmented separately
#' - iVAEs3: Full spatio-temporal segmentation
#' 
#' The autoregressive component incorporates temporal dependencies by using previous time points 
#' as additional conditioning information.
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
#' # Example with STFDF object from spacetime package
#' library(spacetime)
#' library(sp)
#' sp <- unique(coords_time[, 1:2])
#' row.names(sp) <- paste("point", 1:nrow(sp), sep="")
#' library(sp)
#' sp <- SpatialPoints(sp)
#' time <- as.POSIXct("2025-02-01")+3600*(1:n_time)
#' obs_data <- as.data.frame(obs_data)
#' stfdf <- STFDF(sp, time, obs_data) 
#' 
#' # Run iVAEar_radial directly on STFDF object
#' resiVAE_stfdf <- iVAEar_segmentation(
#'   data = stfdf,
#'   latent_dim = 3,
#'   segment_sizes = c(0.1, 0.1, 5), 
#'   joint_segment_inds = c(1, 1, 2),
#'   epochs = 10,
#'   batch_size = 64
#' )
#'
#' @seealso \code{\link{iVAEar}}
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' @export
#' 
iVAEar_segmentation <- function(data, ...) {
  UseMethod("iVAEar_segmentation")
}

#' @rdname iVAEar_radial
#' @export
iVAEar_segmentation.default <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim, n_s,
    time_dim = NULL, seasonal_period = NULL, max_season = NULL, week_component = FALSE, 
    ar_order = 1, epochs, batch_size, ...) {
    n <- dim(data)[1]
    order_inds <- order(locations[, 3], locations[, 1], locations[, 2])
    original_order <- order(order_inds)
    data_ord <- data[order_inds, ]
    aux_data_obj <- form_aux_data_spatial(locations, segment_sizes, 
        joint_segment_inds, time_dim, seasonal_period, max_season, week_component)
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
        ar_order = ar_order, batch_size = batch_size, epochs = epochs, ...)
    class(resVAE) <- c("iVAEar_segmentation", class(resVAE))
    resVAE$spatial_dim <- dim(locations)[2]
    resVAE$locations <- locations
    resVAE$segment_sizes <- segment_sizes
    resVAE$joint_segment_inds <- joint_segment_inds
    resVAE$min_coords <- aux_data_obj$min_coords
    resVAE$max_coords <- aux_data_obj$max_coords
    resVAE$seasonal_period <- seasonal_period
    resVAE$max_season <- aux_data_obj$max_season
    resVAE$week_component <- week_component
    resVAE$time_dim <- time_dim
    resVAE$unique_labels <- aux_data_obj$unique_labels
    resVAE$max_label_per_group <- aux_data_obj$max_label_per_group
    return(resVAE)
}

#' @rdname iVAEar_segmentation
#' @export
iVAEar_segmentation.STFDF <- function(data, segment_sizes, joint_segment_inds, latent_dim, n_s = length(data@sp),
    var_names = NULL, ar_order = 1,
    seasonal_period = NULL, max_season = NULL,
    week_component = FALSE, spatial_kernel = "gaussian", epochs, batch_size, ...) {
    
    spatial_dim <- ncol(sp::coordinates(data))
    spat_coord_names <- colnames(sp::coordinates(data))
    data_df <- as.data.frame(data)
    spatial_locations <- as.matrix(data_df[, spat_coord_names])
    time_points <- as.numeric(data_df[, "timeIndex"])
    locations <- cbind(spatial_locations, time_points)
    if (!is.null(var_names)) {
        data_matrix <- data_df[, var_names]
    } else {
        data_matrix <- as.matrix(data@data)
    }

    # Call the default method with extracted components
    result <- iVAEar_segmentation.default(
        data = data_matrix,
        locations = spatial_locations,
        segment_sizes = segment_sizes,
        joint_segment_inds = joint_segment_inds,
        latent_dim = latent_dim,
        n_s = n_s,
        spatial_dim = spatial_dim,
        ar_order = ar_order,
        seasonal_period = seasonal_period,
        max_season = max_season,
        week_component = week_component,
        spatial_kernel = spatial_kernel,
        epochs = epochs,
        batch_size = batch_size,
        ...
    )
    
    # Add original STFDF data for reference
    result$original_stfdf <- data
    
    return(result)
}

