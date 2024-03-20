#' Radial Basis Function Based Spatio Temporal Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder (iVAE) which uses
#' spatial and temporal radial basis functions as auxiliary data.
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#' @inheritDotParams iVAE -aux_data
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param locations A matrix containing spatial or spatio-temporal locations.
#' @param latent_dim A latent dimension for iVAE.
#' @param epochs A number of epochs to train the model.
#' @param batch_size A batch size.
#' @return
#' An object of class iVAEcoords, inherits from class iVAE.
#' Additionally, the object has the following properties:
#' \item{location_mins}{A vector containing the minimum
#' locations of each dimension.}
#' \item{location_maxs}{A vector containing the maximum
#' locations of each dimension.}
#' \item{spatial_dim}{The spatial dimension.}
#' For more details, see \code{\link{iVAE}}.
#' @details
#' The method uses the min-max normalized coordinates directly as the auxiliary
#' data.
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
#' cor(obs_data, latent_data)
#'
#' # For better peformance, increase the number of epochs.
#' resiVAE <- iVAE_coords(obs_data, coords_time, p,
#'     epochs = 10, batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @export
iVAE_coords <- function(data, locations, latent_dim, epochs, batch_size, ...) {
    location_mins <- apply(locations, 2, min)
    locations_zero <- sweep(locations, 2, location_mins, "-")
    location_maxs <- apply(locations_zero, 2, max)
    locations_norm <- sweep(locations_zero, 2, location_maxs, "/")
    aux_data <- locations_norm

    resVAE <- iVAE(data, aux_data, latent_dim, epochs = epochs, batch_size = batch_size, ...)
    class(resVAE) <- c("iVAEcoords", class(resVAE))
    resVAE$spatial_dim <- dim(locations)[2]
    resVAE$location_mins <- location_mins
    resVAE$location_maxs <- location_maxs
    return(resVAE)
}
