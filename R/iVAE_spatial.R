#' Spatial Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder (iVAE) using the input data
#' and the segmented spatial domain as auxiliary data.
#' @importFrom Rdpack reprompt
#' @inheritDotParams iVAE -aux_data
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param locations A matrix with spatial locations.
#' @param segment_sizes A vector providing sizes for segments.
#' The dimension should match the spatial dimenstion.
#' @param joint_segment_inds A vector indicating which segments
#' are considered jointly. See more in details.
#' @param latent_dim A latent dimension for iVAE.
#' @param epochs Integer specifying the number of training epochs for the iVAE model.
#' @param batch_size Integer specifying the batch size for training the iVAE model.
#' @param ... Additional arguments passed to the underlying \code{\link{iVAEar1}} function.
#' @return
#' An object of class \code{iVAEspatial}, inherits from class \code{iVAE}.
#' Additionally, the object has a property
#' \code{spatial_dim} which gives the dimension of the given locations.
#' For more details, see \code{\link{iVAE}}.
#' @details
#' The method creates the auxiliary data as spatial segments based on
#' the given input parameters.
#' The vector \code{segment_sizes} defines the size of the segments for
#' each spatial dimension separately. The segmentation is then created
#' based on the \code{joint_segment_inds}, which defines dimensions are
#' considered jointly. For example, \code{joint_segment_inds = c(1, 1)}
#' for two dimensional spatial data, defines that the dimensions are
#' considered jointly. This means that the auxiliary variable is vector
#' giving the two dimensional segment that the observations belongs in.
#' For \code{joint_segment_inds = c(1, 2)}, the auxiliary variable would
#' consist of two vectors, first giving one dimensional segment that
#' the observation belongs in the first spatial dimension, and the second
#' giving one dimensional segment that the observation belongs in the
#' secoun spatial dimension. All dimensions are considered jointly as
#' default.
#'
#' After the segmentation, the method calls the function \code{iVAE}
#' using the created auxiliary variables.
#'
#' @references \insertAllCited{}
#' @author Mika Sipilä
#' @seealso
#' \code{\link{iVAE}}
#' \code{\link{generate_nonstationary_spatial_data_by_segments}}
#' @examples
#' n <- 1000
#' coords <- matrix(runif(1000 * 2, 0, 1), ncol = 2)
#' p <- 3
#' # Generate artificial latent data
#' latent_data <- generate_nonstationary_spatial_data_by_segments(
#'     n, p,
#'     coords, 10
#' )$data
#' # Generate artificial observed data by applying a nonlinear mixture
#' obs_data <- mix_data(latent_data, 2)
#' cor(obs_data, latent_data)
#'
#' # For better peformance, increase the number of epochs.
#' resiVAE <- iVAE_spatial(obs_data, coords, c(0.1, 0.1), c(1, 1), 3,
#'     epochs = 10, batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @export
iVAE_spatial <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim, batch_size, epochs, ...) {
    aux_data <- form_aux_data_spatial(locations, segment_sizes, joint_segment_inds)
    resVAE <- iVAE(data, aux_data, latent_dim, batch_size = batch_size, epochs = epochs, ...)
    class(resVAE) <- c("iVAEspatial", class(resVAE))
    resVAE$spatial_dim <- dim(locations)[2]
    return(resVAE)
}
