# Graph specifications for graph-based methods (GiVAE, aux_laplacian).
#
# A graph lives on `n_nodes` nodes, but the data have n observations. The
# layout says which node each observation sits on, and which observations
# are "in the same context" (same position in all the *other* dimensions of
# the data, e.g. the same time point for a spatial graph). Neighbours of an
# observation are the observations in the same context that sit on
# neighbouring nodes.
#
# Layout by `each`: the observations are a regular grid in which the nodes of
# this graph change every `each` rows and cycle through 1..n_nodes. Then
#   node(r)    = ((r - 1) %/% each) %% n_nodes + 1
#   context(r) = (r - 1) %% each + each * ((r - 1) %/% (each * n_nodes)) + 1
# and n must be divisible by n_nodes * each. Examples, rows ordered with the
# spatial unit fastest within time (n = n_s * n_t):
#   spatial  graph (n_s nodes): each = 1    (whole graph repeated n_t times)
#   temporal graph (n_t nodes): each = n_s  (every node repeated n_s times)
#   general graph over all n observations: n_nodes = n, each = 1
#   middle factor of a three-way grid (a fastest, b, c slowest): each = n_a
#
# Layout by `node` (and `context`): explicit integer vectors of length n, for
# unsorted rows or unbalanced data (missing cells simply have fewer
# neighbours). The pair (node, context) must identify an observation.

#' Graph Specification
#'
#' Describes one graph and how its nodes map to the rows of the data. Used by
#' [GiVAE()] and [aux_laplacian()].
#'
#' @param neighborhood A square matrix (nonzero entries of row `i` are the
#'   neighbours of node `i`) or an adjacency list of 1-based neighbour indices,
#'   one element per node. The graph may be directed for the GiVAE prior; the
#'   Laplacian auxiliary variables use the symmetrised graph.
#' @param each Number of consecutive rows sharing a node, see Details. The
#'   default `1` repeats the whole graph over the remaining dimensions.
#' @param node,context Optional integer vectors with one value per row that
#'   give the node of each row and the context (e.g. the time point) in which
#'   neighbours are searched. They replace `each`.
#' @param name Optional label.
#' @details With `each`, the data must form a regular grid: `nrow(x)` is a
#'   multiple of `n_nodes * each`. A spatial graph of `n_s` areas for data
#'   stacked over `n_t` time points uses `each = 1`; a temporal graph of `n_t`
#'   time points for the same data uses `each = n_s`. A graph with one node
#'   per observation uses `each = 1` and no repetition.
#' @return An object of class `nlbss_graph`.
#' @examples
#' n_s <- 4; n_t <- 3
#' W_space <- matrix(0, n_s, n_s); W_space[cbind(1:3, 2:4)] <- 1
#' W_time <- matrix(0, n_t, n_t); W_time[cbind(1:2, 2:3)] <- 1
#' W_space <- pmax(W_space, t(W_space)); W_time <- pmax(W_time, t(W_time))
#' g_space <- nlbss_graph(W_space, name = "space")
#' g_time  <- nlbss_graph(W_time, each = n_s, name = "time")
#' @export
nlbss_graph <- function(neighborhood, each = 1L, node = NULL, context = NULL,
                        name = NULL) {
  adj <- as_adj_list(neighborhood)
  each <- as.integer(each)
  if (length(each) != 1L || is.na(each) || each < 1L) {
    stop("`each` must be a positive integer.", call. = FALSE)
  }
  if (!is.null(context) && is.null(node)) {
    stop("`context` can only be given together with `node`.", call. = FALSE)
  }
  if (!is.null(node)) {
    node <- as.integer(node)
    if (anyNA(node) || any(node < 1L | node > length(adj))) {
      stop("`node` must contain values between 1 and the number of nodes.",
           call. = FALSE)
    }
    if (!is.null(context) && length(context) != length(node)) {
      stop("`context` and `node` must have the same length.", call. = FALSE)
    }
  }
  structure(list(adj_list = adj, n_nodes = length(adj), each = each,
                 node = node, context = context, name = name),
            class = "nlbss_graph")
}

#' @export
print.nlbss_graph <- function(x, ...) {
  deg <- lengths(x$adj_list)
  cat("Graph", if (!is.null(x$name)) paste0("'", x$name, "' "), "with ",
      x$n_nodes, " nodes (mean degree ", format(mean(deg), digits = 3),
      ", max ", max(deg), "); ",
      if (is.null(x$node)) paste0("each = ", x$each) else "explicit node layout",
      "\n", sep = "")
  invisible(x)
}

#' Temporal Graph
#'
#' Convenience constructor for the graph of a time axis: time point `t` has the
#' previous `lags` time points as neighbours, so the GiVAE prior mean of a row
#' is built from the same unit's earlier encodings, as in an autoregressive
#' model. The first time points have fewer neighbours (none for `t = 1`).
#'
#' @param n_time Number of distinct time points (nodes).
#' @param each Number of consecutive rows sharing a time point. For data
#'   stacked as `n_s` units within each time point this is `n_s`.
#' @param lags Number of previous time points used as neighbours.
#' @param node,context Optional explicit layout for unsorted data: the time
#'   point of each row and, as context, the unit it belongs to. See
#'   [nlbss_graph()].
#' @param name Optional label.
#' @param directed If `FALSE` the following time points are neighbours too.
#'   The Laplacian auxiliary variables use the symmetrised graph either way.
#' @return An object of class `nlbss_graph`.
#' @examples
#' nlbss_time_graph(100, each = 20)   # 20 areas observed at 100 time points
#' @export
nlbss_time_graph <- function(n_time, each = 1L, lags = 1L, directed = TRUE,
                             node = NULL, context = NULL, name = "time") {
  lags <- as.integer(lags)
  if (length(lags) != 1L || is.na(lags) || lags < 1L) {
    stop("`lags` must be a positive integer.", call. = FALSE)
  }
  adj <- lapply(seq_len(n_time), function(t) {
    nb <- t - seq_len(lags)
    if (!directed) nb <- c(nb, t + seq_len(lags))
    nb[nb >= 1L & nb <= n_time]
  })
  nlbss_graph(adj, each = each, node = node, context = context, name = name)
}

as_adj_list <- function(g) {
  if (is.list(g) && !is.data.frame(g)) {
    adj <- lapply(g, function(v) as.integer(v))
  } else {
    W <- as.matrix(g)
    if (nrow(W) != ncol(W)) stop("A graph matrix must be square.", call. = FALSE)
    adj <- lapply(seq_len(nrow(W)), function(i) {
      nb <- which(W[i, ] != 0)
      as.integer(nb[nb != i])
    })
  }
  n <- length(adj)
  if (n < 2L) stop("A graph needs at least two nodes.", call. = FALSE)
  if (all(lengths(adj) == 0L)) stop("The graph has no edges.", call. = FALSE)
  for (i in seq_len(n)) {
    nb <- adj[[i]]
    if (anyNA(nb) || any(nb < 1L | nb > n)) {
      stop("Neighbour indices must be between 1 and the number of nodes.",
           call. = FALSE)
    }
    adj[[i]] <- unique(nb[nb != i])
  }
  adj
}

# One graph, a list of graphs, or a matrix / adjacency list (-> one graph with
# the default layout) -> list of nlbss_graph objects.
as_graph_list <- function(x) {
  if (inherits(x, "nlbss_graph")) return(list(x))
  if (is.matrix(x) || inherits(x, "Matrix")) return(list(nlbss_graph(x)))
  if (is.list(x) && length(x) > 0L) {
    first <- x[[1]]
    # an adjacency list has integer vectors as elements; a list of graphs has
    # graph objects, matrices or adjacency lists
    multiple <- inherits(first, "nlbss_graph") || is.matrix(first) ||
      inherits(first, "Matrix") || is.list(first)
    if (multiple) {
      return(lapply(x, function(g) if (inherits(g, "nlbss_graph")) g else nlbss_graph(g)))
    }
    return(list(nlbss_graph(x)))
  }
  stop("`graphs` must be a graph specification, a matrix, an adjacency list ",
       "or a list of these.", call. = FALSE)
}

# Node and context of each of the n rows: list(node, context, n_context).
graph_layout <- function(graph, n) {
  n_g <- graph$n_nodes
  if (!is.null(graph$node)) {
    if (length(graph$node) != n) {
      stop("The explicit `node` layout has length ", length(graph$node),
           " but the data have ", n, " rows.", call. = FALSE)
    }
    node <- graph$node
    context <- if (is.null(graph$context)) rep(1L, n) else {
      match(graph$context, unique(graph$context))
    }
    if (anyDuplicated(cbind(node, context))) {
      stop("Several rows have the same node and context. Give `context` ",
           "(for example the time point) so that each row is identified.",
           call. = FALSE)
    }
  } else {
    e <- graph$each
    if (n %% (n_g * e) != 0L) {
      stop("The number of rows (", n, ") must be a multiple of the number of ",
           "nodes times `each` (", n_g, " x ", e, ") for graph ",
           if (!is.null(graph$name)) paste0("'", graph$name, "'") else "",
           ".", call. = FALSE)
    }
    r <- seq_len(n) - 1L
    node <- (r %/% e) %% n_g + 1L
    context <- r %% e + e * (r %/% (e * n_g)) + 1L
  }
  list(node = as.integer(node), context = as.integer(context),
       n_context = max(context))
}

# Row indices (observations) of the neighbours of every observation.
graph_neighbors <- function(graph, n) {
  lay <- graph_layout(graph, n)
  n_g <- graph$n_nodes
  row_of <- integer(n_g * lay$n_context)
  row_of[(lay$context - 1L) * n_g + lay$node] <- seq_len(n)
  adj <- graph$adj_list
  lapply(seq_len(n), function(i) {
    nb <- adj[[lay$node[i]]]
    if (length(nb) == 0L) return(integer(0))
    r <- row_of[(lay$context[i] - 1L) * n_g + nb]
    r[r > 0L]
  })
}

# Padded neighbour index of all graphs: an n x (G * M) integer matrix (graph g
# occupies columns (g - 1) * M + 1, ..., g * M; NA = no neighbour).
build_neighbor_index <- function(graphs, n, max_neighbors = NULL) {
  nbs <- lapply(graphs, graph_neighbors, n = n)
  if (!is.null(max_neighbors)) {
    cap <- as.integer(max_neighbors)
    nbs <- lapply(nbs, function(nb) {
      lapply(nb, function(v) if (length(v) > cap) sort(sample(v, cap)) else v)
    })
  }
  G <- length(graphs)
  M <- max(1L, max(vapply(nbs, function(nb) max(lengths(nb)), integer(1))))
  index <- matrix(NA_integer_, n, G * M)
  for (g in seq_len(G)) {
    lens <- lengths(nbs[[g]])
    if (sum(lens) == 0L) next
    index[cbind(rep(seq_len(n), lens), (g - 1L) * M + sequence(lens))] <-
      unlist(nbs[[g]], use.names = FALSE)
  }
  list(index = index, G = G, M = M,
       degree = vapply(nbs, function(nb) lengths(nb), integer(n)))
}

# Prior mean of the graph prior, computed from the encoder means Z:
#   rho_max * rho(u_i) * sum_g beta_g(u_i) * mean_{j in N_g(i)} z_j
graph_prior_mean <- function(prior, Z, neigh, rho_max) {
  n <- nrow(Z)
  agg <- matrix(0, n, ncol(Z))
  for (g in seq_len(neigh$G)) {
    num <- matrix(0, n, ncol(Z))
    cnt <- numeric(n)
    for (k in seq_len(neigh$M)) {
      j <- neigh$index[, (g - 1L) * neigh$M + k]
      ok <- which(!is.na(j))
      num[ok, ] <- num[ok, , drop = FALSE] + Z[j[ok], , drop = FALSE]
      cnt[ok] <- cnt[ok] + 1
    }
    agg <- agg + prior$beta[, g] * num / pmax(cnt, 1)
  }
  rho_max * drop(prior$rho) * agg
}