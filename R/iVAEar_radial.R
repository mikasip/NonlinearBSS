#' iVAEar with Radial Basis Functions for Spatial-Temporal Data
#'
#' Fits an identifiable Variational Autoencoder with an autoregressive prior (iVAEar) to spatio-temporal data. 
#' The method constructs the auxiliary data based on spatial and temporal radial basis functions.
#'
#' @param data A matrix of observed data (n x p), where each row is an observation and each column is a feature.
#'   Alternatively, a spatial-temporal data object (e.g., STFDF from spacetime package).
#' @param spatial_locations A matrix of spatial locations corresponding to each observation in \code{data}. Not needed if \code{data} is an STFDF object.
#' @param time_points A vector of time points corresponding to each observation in \code{data}. Not needed if \code{data} is an STFDF object.
#' @param latent_dim The latent dimension of iVAE.
#' @param n_s The number of unique spatial locations in the data. For STFDF objects, defaults to the number of spatial points.
#' @param var_names For STFDF objects only. A character vector specifying which variables to use from the STFDF data. 
#'   If NULL, all variables will be used.
#' @param elevation_var For STFDF objects only. The name of the variable in the STFDF object that contains elevation data.
#'   If NULL, no elevation data will be used.
#' @param elevation Optional vector of elevations corresponding to each observation in \code{data}.
#'   Default is \code{NULL}. Not needed if using STFDF object with \code{elevation_var} specified.
#' @param spatial_dim An integer specifying the number of spatial dimensions. Default is 2 (e.g., latitude and longitude).
#'   For STFDF objects, this is automatically determined from the data.
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
#' This function implements the radial basis function based identifiable VAE with autoregressive components (iVAEr).
#' 
#' In radial basis function based iVAE, the auxiliary variable is defined using radial basis functions
#' \insertCite{hastie2009elements}{NonlinearBSS}. With a large number of appropriate radial basis
#' functions, the model incorporates more spatio-temporal information than by using the coordinates 
#' only. This approach transforms spatial and temporal locations separately into radial basis 
#' functions, similar to recent methods \insertCite{chen2020deepkriging,spatiotemporal_deepkriging}{NonlinearBSS}.
#' 
#' The spatial and temporal radial basis functions are given as:
#' \mjeqn{v^\mathcal{A}(\mathbf{s}; \zeta, \mathbf{o}^\mathcal{S}_i) = v(\| \mathbf{s} - \mathbf{o}^\mathcal{S}_i \|/\zeta)}{v^S(s; zeta, o^S_i) = v(|| s - o^S_i ||/zeta)}{ascii}
#' and
#' \mjeqn{v^\mathcal{T}(t; \zeta, o^\mathcal{T}_i) = v(| t - o^\mathcal{T}_i |/\zeta)}{v^T(t; zeta, o^T_i) = v(| t - o^T_i |/zeta)}{ascii}
#' 
#' where \mjeqn{v}{ascii} is a kernel function such as the Gaussian kernel \mjeqn{v_G(d)=e^{-d^2}}{v_G(d)=exp(-d^2)}{ascii}, 
#' or one of the Wendland kernels \insertCite{wendlandkernel1995}{NonlinearBSS}.
#' 
#' The function uses a multi-resolution approach to form spatial and temporal radial basis functions. 
#' Each resolution level has its own number of evenly spaced node points and scaling parameter:
#' 
#' - Low-level resolution (small number of node points, large scaling parameter): Captures large-scale 
#'   spatial or temporal dependencies
#' 
#' - High-level resolution (many node points, small scaling parameter): Finds finer details of the 
#'   dependence structure
#' 
#' Spatial locations and temporal points are preprocessed to range [0, 1] using min-max normalization.
#' For an H-level spatial resolution, a grid of node points is formed with spacing 1/H and offset 
#' 1/(H+2). Similarly, a G-level temporal resolution uses evenly spaced one-dimensional node points
#' with spacing 1/G and offset 1/(G+2).
#' 
#' The scaling parameters used are:
#' 
#' - Spatial: \mjeqn{\zeta_H = \frac{1}{2.5H}}{zeta_H = 1/(2.5*H)}{ascii}
#' 
#' - Temporal: \mjeqn{\zeta_G = \frac{|o^{\mathcal{T}}_1 - o^{\mathcal{T}}_2|}{\sqrt{2}}}{zeta_G = |o^T_1 - o^T_2|/sqrt(2)}{ascii}
#' 
#' Multiple spatial and temporal resolution levels are used to capture both large-scale and
#' finer dependencies. The advantage of using radial basis functions as auxiliary variables 
#' is that iVAE's auxiliary function provides smooth spatio-temporal trend and variance functions, 
#' which can be used for further analysis such as prediction.
#' 
#' The autoregressive component incorporates temporal dependencies by using previous time points 
#' as additional conditioning information.
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
#'   \item{original_stfdf}{For STFDF objects, the original input STFDF object is stored here.}
#'
#' @examples
#' # Example with standard inputs (matrix data with coordinates)
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
#' resiVAE_stfdf <- iVAEar_radial(
#'   data = stfdf,
#'   latent_dim = 3,
#'   epochs = 10,
#'   batch_size = 64
#' )
#' @references \insertAllCited{}
#' @seealso 
#' \code{\link{iVAEar}}
#' \code{\link{iVAEar_segmentation}}
#' @export
iVAEar_radial <- function(data, ...) {
  UseMethod("iVAEar_radial")
}

#' @rdname iVAEar_radial
#' @export
iVAEar_radial.default <- function(data, spatial_locations, time_points, latent_dim, n_s,
    aux_data = NULL, elevation = NULL, spatial_dim = 2, spatial_basis = c(2, 9),
    temporal_basis = c(9, 17, 37), ar_order = 1, elevation_basis = NULL, 
    seasonal_period = NULL, max_season = NULL,
    week_component = FALSE, spatial_kernel = "gaussian", epochs, batch_size, ...) {
    n <- dim(data)[1]

    order_inds <- do.call(order, as.data.frame(cbind(time_points, spatial_locations)))
    original_order <- order(order_inds)
    data_ord <- data[order_inds, ]
    aux_data_obj <- form_radial_aux_data(spatial_locations, time_points, elevation, spatial_dim, spatial_basis, temporal_basis, elevation_basis, seasonal_period, max_season, spatial_kernel, week_component)
    if (!is.null(aux_data)) {
        aux_data_locs <- apply(aux_data, 2, mean)
        aux_data_sds <- apply(aux_data, 2, sd)
        aux_data <- sweep(aux_data, 2, aux_data_locs, "-")
        aux_data <- sweep(aux_data, 2, aux_data_sds, "/")
        aux_data <- cbind(aux_data, aux_data_obj$aux_data)
    } else {
        aux_data_locs <- NULL
        aux_data_sds <- NULL
        aux_data <- aux_data_obj$aux_data
    }
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
    resVAE$seasonal_period <- seasonal_period
    resVAE$max_season <- aux_data_obj$max_season
    resVAE$min_season <- aux_data_obj$min_season
    resVAE$aux_data_locs <- aux_data_locs
    resVAE$aux_data_sds <- aux_data_sds
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

#' @rdname iVAEar_radial
#' @export
iVAEar_radial.STFDF <- function(data, latent_dim, n_s = length(data@sp),
    var_names = NULL, elevation_var = NULL, spatial_basis = c(2, 9),
    temporal_basis = c(9, 17, 37), ar_order = 1, elevation_basis = NULL, 
    seasonal_period = NULL, max_season = NULL,
    week_component = FALSE, spatial_kernel = "gaussian", epochs, batch_size, ...) {
    
    spatial_dim <- ncol(sp::coordinates(data))
    spat_coord_names <- colnames(sp::coordinates(data))
    data_df <- as.data.frame(data)
    spatial_locations <- as.matrix(data_df[, spat_coord_names])
    time_points <- as.numeric(data_df[, "timeIndex"])
    if (!is.null(var_names)) {
        data_matrix <- data_df[, var_names]
    } else {
        data_matrix <- as.matrix(data@data)
    }
    
    if (!is.null(elevation_var)) {
        elevation <- as.numeric(data_df[, elevation_var])
    } else {
        elevation <- NULL
    }

    # Call the default method with extracted components
    result <- iVAEar_radial.default(
        data = data_matrix,
        spatial_locations = spatial_locations,
        time_points = time_points,
        latent_dim = latent_dim,
        n_s = n_s,
        elevation = elevation,
        spatial_dim = spatial_dim,
        spatial_basis = spatial_basis,
        temporal_basis = temporal_basis,
        ar_order = ar_order,
        elevation_basis = elevation_basis,
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
