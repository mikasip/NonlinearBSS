#' Spatial Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder (iVAE) using the input data
#' and the segmented spatial domain as auxiliary data.
#' @import tensorflow
#' @import keras
#' @importFrom Rdpack reprompt
#' @inheritDotParams iVAE -aux_data
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param locations A matrix with spatial locations.
#' @param segment_sizes A vector providing sizes for segments.
#' The dimension should match the spatial dimenstion.
#' @param joint_segment_inds A vector indicating which segments.
#' are considered jointly. See more in details.
#' @param latent_dim A latent dimension for iVAE.
#' @param test_inds A vector giving the indices of the data, which
#' are used as a test data.
#' @return
#' An object of class iVAESpatial, inherits from class iVAE.
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
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim,
    test_inds = NULL, ...) {
    n <- nrow(data)
    location_mins <- apply(locations, 2, min)
    locations_zero <- sweep(locations, 2, location_mins, "-")
    location_maxs <- apply(locations_zero, 2, max)
    aux_data <- matrix(nrow = n, ncol = 0)
    for (i in unique(joint_segment_inds)) {
        inds <- which(joint_segment_inds == i)
        labels <- rep(0, n)
        lab <- 1
        loop_dim <- function(j, sample_inds) {
            ind <- inds[j]
            seg_size <- segment_sizes[ind]
            seg_limits <- seq(0, (location_maxs[ind]), seg_size)
            for (coord in seg_limits) {
                cur_inds <- which(locations[sample_inds, ind] >= coord &
                    locations[sample_inds, ind] < coord + seg_size)
                cur_sample_inds <- sample_inds[cur_inds]
                if (j == length(inds)) {
                    labels[cur_sample_inds] <<- lab
                    lab <<- lab + 1
                } else {
                    loop_dim(j + 1, cur_sample_inds)
                }
            }
        }
        loop_dim(1, 1:n)
        labels <- as.numeric(as.factor(labels)) # To ensure that
        # empty segments are reduced
        aux_data <- cbind(aux_data, model.matrix(~ 0 + as.factor(labels)))
    }
    test_data <- NULL
    test_aux_data <- NULL
    if (!is.null(test_inds)) {
        test_data <- data[test_inds, ]
        test_aux_data <- aux_data[test_inds, ]
    }
    resVAE <- iVAE(data, aux_data, latent_dim,
        test_data = test_data,
        test_data_aux = test_aux_data, ...
    )
    class(resVAE) <- c("iVAEspatial", class(resVAE))
    resVAE$spatial_dim <- dim(locations)[2]
    resVAE$test_data <- test_data
    resVAE$test_aux_data <- test_aux_data
    return(resVAE)
}
