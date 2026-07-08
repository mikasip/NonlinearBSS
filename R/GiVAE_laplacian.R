
# ------------------------------------------------------------------------------
# User-friendly wrapper: compute Laplacian eigenvectors and call GiVAE
# ------------------------------------------------------------------------------
 
#' GiVAE with Laplacian Eigenvector Auxiliary Data
#'
#' @description
#' Convenience wrapper that computes the first \code{K} non-trivial
#' eigenvectors of the graph Laplacian from \code{adj_list} and uses them as
#' auxiliary data \eqn{u_i} before calling \code{\link{GiVAE}}.
#' The eigenvectors encode each area's position in the graph topology and
#' satisfy the iVAE injectivity condition required for identifiability.
#'
#' @param data      N x P matrix of observed data.
#' @param adj_list  List of length N of 1-based integer neighbor index vectors.
#' @param latent_dim Integer number of latent components.
#' @param K Integer, number of Laplacian eigenvectors to use. Default 10.
#' @param epochs,batch_size Training parameters.
#' @param ... Additional arguments forwarded to \code{\link{GiVAE}}.
#'
#' @return An object of class \code{c("GiVAElaplacian", "GiVAE")} with
#'   additional fields \code{phi} (N x K eigenvector matrix) and
#'   \code{lambda} (K eigenvalues).
#' @export
GiVAE_laplacian <- function(
    data,
    adj_list,
    latent_dim,
    K          = 10,
    epochs,
    batch_size,
    ...
) {
  eig_result <- compute_laplacian_eigenvectors(adj_list, K = K)
  aux_data   <- eig_result$phi   # N x K
 
  result         <- GiVAE(data, aux_data, latent_dim, adj_list,
                          epochs = epochs, batch_size = batch_size, ...)
  result$K       <- K
  result$lambda  <- eig_result$lambda
  result$phi     <- aux_data
  class(result)  <- c("GiVAElaplacian", "GiVAE")
  return(result)
}