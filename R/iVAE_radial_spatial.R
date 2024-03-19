#' Radial Basis Function Based Spatial Identifiable Variational Autoencoder
#' @description Trains an identifiable variational autoencoder (iVAE) using the input data
#' and spatial radial basis functions as auxiliary data.
#' @import tensorflow
#' @import keras
#' @import rdist
#' @importFrom Rdpack reprompt
#' @inheritDotParams iVAE -aux_data
#' @param data A matrix with P columns and N rows containing the observed data.
#' @param locations A matrix with spatial locations.
#' @param latent_dim A latent dimension for iVAE.
#' @param num_basis A vector containing the number of basis functions for
#' each resolution. Default value is (2, 9).
#' @param kernel A kernel function to be used to form the radial basis
#' functions. Either \code{"gaussian"} (default) or \code{"wendland"}.
#' @param test_inds An optional vector of the indices of the rows used as
#' test data.
#' @return
#' An object of class iVAEradial, inherits from class iVAE.
#' Additionally, the object has the following properties:
#' \code{min_coords} A vector of minimum coordinates.
#' \code{max_coords} A vector of maximum coordinates.
#' \code{num_basis} Same as \code{num_basis} in the function call.
#' \code{spatial_dim} The dimension the given locations.
#' For more details, see \code{\link{iVAE}}.
#' @details
#' The method creates the auxiliary data as radial basis functions based on
#' the given input parameters.
#' The vector \code{num_basis} defines the resolution levels used to create the
#' spatial radial basis functions. For example \code{num_basis = c(2, 9)}
#' creates the radial basis functions with resolution levels 2 and 9. The
#' radial basis functions are created identically as in
#' \insertCite{chen2020deepkriging}{NonlinearBSS}.
#'
#' After the forming the radial basis functions, the method
#' calls the function \code{iVAE} using the created auxiliary variables.
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
#' resiVAE <- iVAE_radial_spatial(obs_data, coords, c(0.1, 0.1), c(1, 1), 3,
#'     epochs = 10, batch_size = 64
#' )
#' cormat <- cor(resiVAE$IC, latent_data)
#' cormat
#' absolute_mean_correlation(cormat)
#'
#' @export
iVAE_radial_spatial <- function(data, locations, latent_dim, num_basis = c(2, 9), kernel = "gaussian", test_inds = NULL, epochs, batch_size, ...) {
    kernel <- match.arg(kernel, c("gaussian", "wendland"))
    spatial_dim <- dim(locations)[2]
    N <- dim(data)[1]
    min_coords <- apply(locations, 2, min)
    locations_new <- sweep(locations, 2, min_coords, "-")
    max_coords <- apply(locations_new, 2, max)
    locations_new <- sweep(locations_new, 2, max_coords, "/")
    knots_1d <- sapply(num_basis, FUN = function(i) seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i))
    phi_all <- matrix(0, ncol = 0, nrow = N)
    for (i in seq_along(num_basis)) {
        theta <- 1 / num_basis[i] * 2.5
        knot_list <- replicate(spatial_dim, knots_1d[[i]], simplify = FALSE)
        knots <- as.matrix(expand.grid(knot_list))
        phi <- cdist(locations_new, knots) / theta
        dist_leq_1 <- phi[which(phi <= 1)]
        dist_g_1_ind <- which(phi > 1)
        phi[which(phi <= 1)] <- switch(kernel,
            gaussian = gaussian_kernel(dist_leq_1),
            wendland = wendland_kernel(dist_leq_1)
        )
        phi[dist_g_1_ind] <- 0
        phi_all <- cbind(phi_all, phi)
    }
    aux_data <- phi_all
    if (!is.null(test_inds)) {
        test_data <- data[test_inds, ]
        train_data <- data[-test_inds, ]
        test_aux_data <- data[test_inds, ]
        train_aux_data <- data[-test_inds, ]
    } else {
        test_data <- NULL
        train_data <- data
        test_aux_data <- NULL
        train_aux_data <- aux_data
    }
    resVAE <- iVAE(train_data, train_aux_data, latent_dim, test_data = test_data, test_data_aux = test_aux_data, epochs = epochs, batch_size = batch_size, get_prior_means = FALSE, ...)
    class(resVAE) <- c("iVAEradial", class(resVAE))
    resVAE$min_coords <- min_coords
    resVAE$max_coords <- max_coords
    resVAE$num_basis <- num_basis
    # resVAE$phi_maxs <- phi_maxs
    resVAE$spatial_dim <- dim(locations)[2]
    return(resVAE)
}
