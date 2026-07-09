
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
#' @param adj_list  Either a single adjacency list (list of length N of 1-based
#' integer neighbor index vectors) or a list of adjacency lists (one per graph),
#' each of which must be length N. When providing multiple adjacency lists,
#' supply \\code{K} as an integer vector giving the number of eigenvectors per
#' graph.
#' @param latent_dim Integer number of latent components.
#' @param K Integer or integer vector. If a single integer and multiple graphs
#' are provided, the same \\code{K} is used for all graphs. If a vector, its
#' length must match the number of adjacency lists; each entry gives the number
#' of eigenvectors to compute for the corresponding graph.
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
  # Allow a single adjacency list or a list of adjacency lists (multiple graphs)
  multiple_graphs <- is.list(adj_list) && length(adj_list) > 0 && is.list(adj_list[[1]])

  if (multiple_graphs) {
    adj_lists <- adj_list
    n_graphs  <- length(adj_lists)
    if (length(K) == 1) K <- rep(as.integer(K), n_graphs)
    if (length(K) != n_graphs) stop("When providing multiple adjacency lists, length(K) must equal number of graphs")

    phi_list    <- vector("list", n_graphs)
    lambda_list <- vector("list", n_graphs)
    eig_list    <- vector("list", n_graphs)
    for (g in seq_len(n_graphs)) {
      eig_res_g <- compute_laplacian_eigenvectors(adj_lists[[g]], K = K[g])
      phi_list[[g]]    <- eig_res_g$phi
      lambda_list[[g]] <- eig_res_g$lambda
      eig_list[[g]]    <- eig_res_g
    }

    # Ensure all phi matrices have the same number of rows (N)
    Ns <- vapply(phi_list, nrow, integer(1))
    if (length(unique(Ns)) != 1L) stop("All adjacency lists must refer to the same number of nodes")
    aux_data <- do.call(cbind, phi_list) # N x sum(K)
    lambda   <- unlist(lambda_list)
    Ks       <- K
  } else {
    eig_result <- compute_laplacian_eigenvectors(adj_list, K = K)
    aux_data   <- eig_result$phi   # N x K
    eig_list   <- list(eig_result)
    lambda     <- eig_result$lambda
    Ks         <- length(lambda)
    n_graphs   <- 1L
  }

  result <- GiVAE(data, aux_data, latent_dim, adj_list,
                  epochs = epochs, batch_size = batch_size, ...)

  result$Ks          <- Ks
  result$lambda      <- lambda
  result$phi         <- aux_data
  result$eig_results <- eig_list
  result$n_graphs    <- if (exists("n_graphs")) n_graphs else 1L
  class(result)      <- c("GiVAElaplacian", "GiVAE")
  return(result)
}