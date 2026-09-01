
#' Batch-wise Radial Basis Function iVAE for Spatio-Temporal Data
#'
#' @description Memory-efficient counterpart of \code{\link{iVAE_radial_spatio_temporal}}.
#' Instead of precomputing the full n x d radial basis function auxiliary matrix in R,
#' this constructs the RBF features batch-wise inside the Keras graph via
#' \code{\link{iVAE_rbf_b}}, using only the raw spatial locations and time points plus
#' the bounding parameters needed to reconstruct the same basis functions.
#'
#' @inheritParams iVAE_radial_spatio_temporal
#' @param ... Additional arguments passed to \code{\link{iVAE_rbf_b}}.
#'
#' @return An object of class \code{iVAEradial_st}, which inherits from class \code{iVAE},
#' structurally equivalent to the output of \code{\link{iVAE_radial_spatio_temporal}}.
#'
#' @seealso \code{\link{iVAE_radial_spatio_temporal}}, \code{\link{iVAE_rbf_b}}
#' @author Mika Sipilä
#' @export
iVAE_radial_spatio_temporal_b <- function(data, spatial_locations, time_points, latent_dim,
    aux_data = NULL, elevation = NULL, spatial_dim = 2, spatial_basis = c(2, 9),
    temporal_basis = c(9, 17, 37), elevation_basis = NULL, seasonal_period = NULL,
    spatial_kernel = "gaussian", week_component = FALSE, epochs, batch_size, ...) {
 
  aux_params <- form_radial_params(
    spatial_locations, time_points, elevation,
    spatial_dim, spatial_basis, temporal_basis,
    elevation_basis, seasonal_period, NULL,
    spatial_kernel, week_component
  )
 
  if (!is.null(aux_data)) {
    aux_data_locs <- apply(aux_data, 2, mean)
    aux_data_sds <- apply(aux_data, 2, sd)
    aux_extra_scaled <- sweep(aux_data, 2, aux_data_locs, "-")
    aux_extra_scaled <- sweep(aux_extra_scaled, 2, aux_data_sds, "/")
  } else {
    aux_data_locs <- NULL
    aux_data_sds <- NULL
    aux_extra_scaled <- NULL
  }
 
  resVAE <- iVAE_rbf_b(
    data = data,
    spatial_locations = spatial_locations,
    time_points = time_points,
    latent_dim = latent_dim,
    aux_extra = aux_extra_scaled,
    spatial_dim = spatial_dim,
    spatial_basis = spatial_basis,
    temporal_basis = temporal_basis,
    spatial_kernel = spatial_kernel,
    week_component = week_component,
    seasonal_period = seasonal_period,
    max_season = aux_params$max_season,
    elevation = elevation,
    elevation_basis = elevation_basis,
    min_coords = aux_params$min_coords,
    max_coords = aux_params$max_coords,
    min_time_point = aux_params$min_time_point,
    max_time_point = aux_params$max_time_point,
    epochs = epochs, batch_size = batch_size, ...
  )
 
  class(resVAE) <- c("iVAEradial_st", class(resVAE))
  resVAE$min_coords <- aux_params$min_coords
  resVAE$max_coords <- aux_params$max_coords
  if (!is.null(seasonal_period)) resVAE$seasonal_period <- seasonal_period
  resVAE$week_component <- week_component
  resVAE$spatial_basis <- spatial_basis
  resVAE$temporal_basis <- temporal_basis
  resVAE$elevation_basis <- elevation_basis
  resVAE$aux_data_locs <- aux_data_locs
  resVAE$aux_data_sds <- aux_data_sds
  resVAE$spatial_kernel <- aux_params$spatial_kernel
  resVAE$min_time_point <- aux_params$min_time_point
  resVAE$max_time_point <- aux_params$max_time_point
  resVAE$max_season <- aux_params$max_season
  resVAE$min_season <- aux_params$min_season
  resVAE$min_elevation <- aux_params$min_elevation
  resVAE$max_elevation <- aux_params$max_elevation
  resVAE$spatial_dim <- spatial_dim
  resVAE$locations <- spatial_locations
  resVAE$time <- time_points
  resVAE$elevation <- elevation
 
  return(resVAE)
}