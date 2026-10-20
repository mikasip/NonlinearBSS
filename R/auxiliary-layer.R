# Batch-wise computation of the auxiliary variables inside the network.
#
# With `batch_aux = TRUE` the auxiliary input of the network contains, for
# each auxiliary specification, a small block of raw inputs (coordinates,
# time points, day of week, elevation, covariates). The auxiliary variables
# are computed from these for each batch by a Keras layer, which avoids
# storing the full n x d auxiliary matrix. The computations mirror
# transform_aux(), and the features are in the same order.

aux_raw_block <- function(spec, dat, values, training) {
  switch(spec$type,
    rbf = {
      check_rbf_data(spec, dat)
      cbind(
        if (spec$use_space) dat$coords,
        if (rbf_needs_time(spec)) dat$time,
        if (spec$week_component) dat$weekday,
        elevation_values(spec, dat, values, training)
      )
    },
    segments = {
      L <- seg_locations(dat)
      if (ncol(L) != spec$dims) {
        stop("The new data have a different number of dimensions than the ",
             "training data.", call. = FALSE)
      }
      cbind(L, if (spec$week_component) dat$weekday)
    },
    coords = seg_locations(dat),
    covariates = covariate_values(spec, dat, values, training),
    moran = stop("Batch-wise auxiliary computation is not supported for Moran eigenvector maps; use `batch_aux = FALSE` or pass a precomputed auxiliary matrix.", call. = FALSE),
    laplacian = stop("Batch-wise auxiliary computation is not supported for Laplacian eigenvectors; use `batch_aux = FALSE`.", call. = FALSE)
  )
}

rbf_needs_time <- function(spec) spec$use_time || !is.null(spec$seasonal_period)

tf_cols <- function(x, idx) {
  if (length(idx) == 1) return(keras3::op_expand_dims(x[, idx], -1L))
  x[, idx[1]:idx[length(idx)]]
}

tf_const <- function(x) keras3::op_convert_to_tensor(x, dtype = "float32")

tf_row <- function(x) tf_const(matrix(x, nrow = 1))

tf_one_hot <- function(x, levels) {
  keras3::op_cast(keras3::op_equal(x, tf_row(levels)), "float32")
}

tf_season_split <- function(t, period, seasons) {
  season <- keras3::op_clip(keras3::op_floor(t / period), min(seasons), max(seasons))
  list(one_hot = tf_one_hot(season, seasons),
       within = t - period * keras3::op_floor(t / period) + 1)
}

# A function mapping a block of raw inputs of one specification to features.
aux_tf_block <- function(spec) {
  switch(spec$type,
    rbf = rbf_tf_block(spec),
    segments = segments_tf_block(spec),
    coords = function(x) (x - tf_row(spec$mins)) / tf_row(spec$ranges),
    covariates = function(x) (x - tf_row(spec$center)) / tf_row(spec$scale_by),
    moran = stop("Batch-wise auxiliary computation is not supported for Moran eigenvector maps; use `batch_aux = FALSE` or pass a precomputed auxiliary matrix.", call. = FALSE),
    laplacian = stop("Batch-wise auxiliary computation is not supported for Laplacian eigenvectors; use `batch_aux = FALSE`.", call. = FALSE)
  )
}

segments_tf_block <- function(spec) {
  d <- spec$dims
  function(x) {
    L <- tf_cols(x, seq_len(d))
    feats <- list()
    season <- NULL
    if (!is.null(spec$seasonal_period)) {
      t <- tf_cols(L, d)
      season <- tf_season_split(t, spec$seasonal_period, spec$seasons)
      L <- keras3::op_concatenate(
        c(if (d > 1) list(tf_cols(L, seq_len(d - 1))), list(season$within)),
        axis = -1L)
    }
    groups <- unique(spec$joint)
    for (k in seq_along(groups)) {
      dims <- which(spec$joint == groups[k])
      cells <- spec$level_cells[[k]]
      hit <- NULL
      for (m in seq_along(dims)) {
        j <- dims[m]
        cell <- keras3::op_floor((tf_cols(L, j) - spec$mins[j]) / spec$size[j])
        eq <- tf_one_hot(cell, cells[, m])
        hit <- if (is.null(hit)) eq else hit * eq
      }
      feats[[k]] <- hit
    }
    if (!is.null(season)) feats[[length(feats) + 1]] <- season$one_hot
    if (spec$week_component) {
      feats[[length(feats) + 1]] <- tf_one_hot(tf_cols(x, d + 1), 0:6)
    }
    keras3::op_concatenate(feats, axis = -1L)
  }
}

rbf_tf_block <- function(spec) {
  pos <- 0
  next_cols <- function(k) {
    idx <- if (k > 0) seq(pos + 1, pos + k) else integer(0)
    pos <<- pos + k
    idx
  }
  space_cols <- next_cols(if (spec$use_space) spec$spatial_dim else 0)
  time_col <- next_cols(if (rbf_needs_time(spec)) 1 else 0)
  week_col <- next_cols(if (spec$week_component) 1 else 0)
  elevation_col <- next_cols(if (!is.null(spec$elevation)) 1 else 0)

  function(inputs) {
    feats <- list()
    if (spec$use_space) {
      s <- (tf_cols(inputs, space_cols) - tf_row(spec$coord_min)) /
        tf_row(spec$coord_range)
      s2 <- keras3::op_sum(keras3::op_square(s), axis = -1L, keepdims = TRUE)
      for (H in spec$spatial_basis) {
        K <- rbf_knots(H, spec$spatial_dim)
        d2 <- s2 + tf_row(rowSums(K^2)) - 2 * keras3::op_matmul(s, tf_const(t(K)))
        d <- keras3::op_sqrt(keras3::op_maximum(d2, 0)) / (2.5 / H)
        feats[[length(feats) + 1]] <- if (spec$kernel == "gaussian") {
          keras3::op_exp(-keras3::op_square(d))
        } else {
          keras3::op_where(d < 1, (1 - d)^6 * (35 * d^2 + 18 * d + 3) / 3,
                           keras3::op_zeros_like(d))
        }
      }
    }
    if (spec$week_component) {
      feats[[length(feats) + 1]] <- tf_one_hot(tf_cols(inputs, week_col), 0:6)
    }
    if (length(time_col)) tt <- tf_cols(inputs, time_col)
    if (!is.null(spec$seasonal_period)) {
      season <- tf_season_split(tt, spec$seasonal_period, spec$seasons)
      feats[[length(feats) + 1]] <- season$one_hot
      tt <- season$within
    }
    if (spec$use_time) {
      for (G in spec$temporal_basis) {
        tk <- temporal_knots(G, spec$time_min, spec$time_max)
        z <- (tt - tf_row(tk$knots)) / tk$kappa
        feats[[length(feats) + 1]] <- keras3::op_exp(-0.5 * keras3::op_square(z))
      }
    }
    if (length(elevation_col)) {
      e <- tf_cols(inputs, elevation_col)
      for (E in spec$elevation_basis) {
        ek <- temporal_knots(E, spec$elevation_min, spec$elevation_max)
        z <- (e - tf_row(ek$knots)) / ek$kappa
        feats[[length(feats) + 1]] <- keras3::op_exp(-0.5 * keras3::op_square(z))
      }
    }
    keras3::op_concatenate(feats, axis = -1L)
  }
}

# The function used by the Keras layer: raw auxiliary input -> features.
aux_tf_function <- function(config) {
  blocks <- lapply(config, function(b) list(cols = b$cols, f = aux_tf_block(b$spec)))
  function(inputs) {
    feats <- lapply(blocks, function(b) b$f(tf_cols(inputs, b$cols)))
    if (length(feats) == 1) feats[[1]] else keras3::op_concatenate(feats, axis = -1L)
  }
}

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