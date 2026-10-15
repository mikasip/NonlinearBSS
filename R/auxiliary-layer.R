# Auxiliary data from graph Laplacian eigenvectors.
#
# The eigenvectors are computed once per graph on the graph's own nodes and
# then copied to the rows of the data according to the graph layout (see
# graph.R), so a spatial graph of n_s areas and a temporal graph of n_t time
# points can be combined for n = n_s * n_t rows. Like the Moran auxiliary
# variables they depend on the whole graph and cannot be computed batch-wise.

#' Auxiliary Data Based on Graph Laplacian Eigenvectors
#'
#' Specifies auxiliary variables formed by the `K` smoothest non-trivial
#' eigenvectors (smallest non-zero Laplacian eigenvalues) of one or several
#' graphs. Each graph contributes its own block of columns, and its
#' eigenvectors are mapped to the observations with the layout of the graph,
#' see [nlbss_graph()].
#'
#' @param graphs A graph specification [nlbss_graph()], a list of them, or a
#'   matrix / adjacency list (one node per row of the data, or repeated over
#'   the data with the default layout).
#' @param K Number of eigenvectors: one value, used for every graph, or one
#'   value per graph.
#' @details The graph is symmetrised (an edge exists if either direction is
#'   given). One zero-eigenvalue eigenvector per connected component is
#'   dropped, since these are constant on components.
#' @return An object of class `aux_laplacian`.
#' @seealso [aux_moran()], [nlbss_graph()], [GiVAE()]
#' @export
aux_laplacian <- function(graphs, K = 10L) {
  graphs <- as_graph_list(graphs)
  K <- as.integer(K)
  if (length(K) == 1L) K <- rep(K, length(graphs))
  if (length(K) != length(graphs) || anyNA(K) || any(K < 1L)) {
    stop("`K` must be a positive integer, or one per graph.", call. = FALSE)
  }
  new_aux("laplacian", graphs = graphs, K = K)
}

# Sparse symmetric binary adjacency matrix without self loops.
graph_sparse_W <- function(adj) {
  n <- length(adj)
  lens <- lengths(adj)
  W <- Matrix::sparseMatrix(i = rep(seq_len(n), lens), j = unlist(adj, use.names = FALSE),
                            x = 1, dims = c(n, n))
  W <- W + Matrix::t(W)
  W@x[] <- 1
  Matrix::diag(W) <- 0
  Matrix::drop0(W)
}

count_components <- function(W) {
  n <- nrow(W)
  adj <- lapply(seq_len(n), function(i) {
    W@i[W@p[i] + seq_len(W@p[i + 1L] - W@p[i])] + 1L   # empty for isolated nodes
  })
  comp <- integer(n)
  k <- 0L
  for (s in seq_len(n)) {
    if (comp[s] != 0L) next
    k <- k + 1L
    comp[s] <- k
    stack <- s
    while (length(stack)) {
      v <- stack[length(stack)]
      stack <- stack[-length(stack)]
      nb <- adj[[v]]
      new <- nb[comp[nb] == 0L]
      comp[new] <- k
      stack <- c(stack, new)
    }
  }
  k
}

# Smallest eigenpairs of a sparse Laplacian by shift-and-invert (Lanczos).
# Returns NULL if RSpectra is missing or does not converge, so that the caller
# can fall back to a dense eigendecomposition. The shift is tiny so that the
# wanted (smallest) eigenvalues are well separated after inversion even when
# they are very close to each other, as on long chains.
sparse_smallest_eigs <- function(L, k) {
  n <- nrow(L)
  if (!requireNamespace("RSpectra", quietly = TRUE)) return(NULL)
  res <- tryCatch(
    RSpectra::eigs_sym(methods::as(L, "CsparseMatrix"), k = k, sigma = -1e-6,
                       opts = list(ncv = min(n, max(2L * k + 1L, 40L)),
                                   maxitr = 5000L)),
    error = function(e) NULL)
  if (is.null(res) || length(res$values) < k || anyNA(res$values)) return(NULL)
  list(values = res$values, vectors = res$vectors)
}

# `dense_max`: graphs up to this size use base eigen() directly.
compute_laplacian_eigenvectors <- function(adj_list, K = 10L, dense_max = 1000L) {
  W <- methods::as(graph_sparse_W(adj_list), "generalMatrix")
  W <- methods::as(W, "CsparseMatrix")
  n <- nrow(W)
  n_comp <- count_components(W)
  K <- min(as.integer(K), n - n_comp)
  if (K < 1L) stop("The graph has too few nodes for the requested K.", call. = FALSE)
  L <- Matrix::Diagonal(n, x = Matrix::rowSums(W)) - W
  eig <- if (n > dense_max) sparse_smallest_eigs(L, K + n_comp)
  if (is.null(eig)) {
    if (n > 5000L) {
      stop("The sparse eigensolver is not available or did not converge for a ",
           "graph with ", n, " nodes. Install 'RSpectra' or supply the ",
           "auxiliary variables as a matrix.", call. = FALSE)
    }
    if (n > dense_max) {
      message("Using a dense eigendecomposition for a graph with ", n, " nodes.")
    }
    eig <- eigen(as.matrix(L), symmetric = TRUE)
  }
  ord <- order(eig$values)
  vals <- eig$values[ord]
  vecs <- eig$vectors[, ord, drop = FALSE]
  keep <- n_comp + seq_len(K)
  phi <- vecs[, keep, drop = FALSE]
  for (k in seq_len(ncol(phi))) {
    if (phi[which.max(abs(phi[, k])), k] < 0) phi[, k] <- -phi[, k]
  }
  list(phi = phi, lambda = vals[keep], n_components = n_comp)
}

#' @export
fit_aux.aux_laplacian <- function(spec, dat) {
  n <- nrow(dat$X)
  res <- Map(function(g, k) compute_laplacian_eigenvectors(g$adj_list, k),
             spec$graphs, spec$K)
  for (g in spec$graphs) graph_layout(g, n)    # checks that the layout fits the data
  for (g in seq_along(res)) {
    colnames(res[[g]]$phi) <- paste0("LAP", g, "_", seq_len(ncol(res[[g]]$phi)))
  }
  spec$phi <- lapply(res, `[[`, "phi")
  spec$lambda <- lapply(res, `[[`, "lambda")   # not `values`: that name is used by covariates
  spec$n_components <- vapply(res, `[[`, integer(1), "n_components")
  spec$n_features <- sum(vapply(spec$phi, ncol, integer(1)))
  spec$fitted <- TRUE
  spec
}

#' @export
transform_aux.aux_laplacian <- function(spec, dat, values, training) {
  if (is.null(spec$phi)) {
    stop("The Laplacian auxiliary variables have not been fitted yet.", call. = FALSE)
  }
  n <- nrow(dat$X)
  blocks <- Map(function(g, phi) phi[graph_layout(g, n)$node, , drop = FALSE],
                spec$graphs, spec$phi)
  do.call(cbind, blocks)
}