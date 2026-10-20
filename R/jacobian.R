# Log-determinant (Jacobian) term of the GiVAE graph prior.
#
# For latent component l the prior is the simultaneous autoregression
#   (I - B_l) z_l = eps_l,  eps_il ~ N(0, sigma_l^2(u_i)),
#   B_l[i, j] = sum_g rho_max * rho_l(u_i) * beta_{g,l}(u_i) * 1[j in N_g(i)] / |N_g(i)|,
# whose exact log density contains log |det(I - B_l)|. The residual terms in
# the ELBO are unchanged; this file adds the missing term
#   sum_l log |det(I - B_l)|
# (divided by the number of rows, to match the per-row loss).
#
# Structure used:
# * log det(I - B) is the sum of log det over the strongly connected
#   components of the observation graph (B is block triangular after ordering
#   the components). Components with one node contribute 0, so a purely
#   acyclic graph (e.g. a directed time graph, or nlbss_dag()) needs nothing.
#   Spatial + directed temporal graphs split into one component per time
#   point.
# * Every training step estimates the sum over components unbiasedly by
#   sampling one component s with probability q_s (proportional to its size)
#   and using log det(I - B_s) / q_s.
# * The log det of the sampled component is computed
#   - "exact": dense LU (slogdet), for components up to `jacobian_max_exact`
#     nodes;
#   - "stochastic": the series  log det(I - B) = - sum_k tr(B^k) / k  with
#     Hutchinson probes and sparse matrix-vector products, truncated after
#     `jacobian_terms` terms. B has spectral radius at most rho_max < 1, so the
#     series converges, but slowly for rho_max close to 1: the truncation
#     under-estimates the penalty of the smooth modes (a bias towards weaker
#     penalisation). Use the exact option or more terms when this matters, and
#     check with givae_logdet().

# Edge list, strongly connected components and padded arrays of the
# observation graph. Returns list(m = 0) if there is nothing to do.
graph_blocks <- function(neigh, n) {
  src <- integer(0); dst <- integer(0); gr <- integer(0); w <- numeric(0)
  for (g in seq_len(neigh$G)) {
    idx <- neigh$index[, (g - 1L) * neigh$M + seq_len(neigh$M), drop = FALSE]
    deg <- rowSums(!is.na(idx))
    hit <- which(!is.na(idx), arr.ind = TRUE)
    if (nrow(hit) == 0L) next
    i <- hit[, 1L]
    src <- c(src, i)
    dst <- c(dst, idx[hit])
    gr <- c(gr, rep(g - 1L, length(i)))
    w <- c(w, 1 / deg[i])
  }
  if (length(src) == 0L) return(list(m = 0L))
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("Package 'igraph' is needed for the Jacobian term of GiVAE.", call. = FALSE)
  }
  net <- igraph::make_graph(as.vector(rbind(src, dst)), n = n, directed = TRUE)
  comp <- igraph::components(net, mode = "strong")$membership
  size <- tabulate(comp)
  ids <- which(size >= 2L)
  m <- length(ids)
  if (m == 0L) return(list(m = 0L))

  cid <- match(comp, ids)                       # NA: trivial component
  nodes <- which(!is.na(cid))
  rows_list <- split(nodes, factor(cid[nodes], levels = seq_len(m)))
  sizes <- lengths(rows_list)
  pos <- integer(n)
  pos[unlist(rows_list, use.names = FALSE)] <- sequence(sizes)   # position in component

  keep <- which(!is.na(cid[src]) & cid[src] == cid[dst])
  ord <- order(cid[src[keep]])
  keep <- keep[ord]
  e_comp <- cid[src[keep]]
  n_e <- tabulate(e_comp, m)
  Emax <- max(n_e)
  Smax <- max(sizes)
  slot <- sequence(n_e)
  fill <- function(vals, pad) {
    M <- matrix(pad, m, Emax)
    M[cbind(e_comp, slot)] <- vals
    M
  }
  rows <- matrix(0L, m, Smax)
  rows[cbind(rep(seq_len(m), sizes), sequence(sizes))] <- unlist(rows_list, use.names = FALSE) - 1L

  edges <- data.frame(src = src[keep], dst = dst[keep], g = gr[keep] + 1L,
                      w = w[keep], comp = e_comp)
  list(m = m, sizes = sizes, Smax = Smax, Emax = Emax, q = sizes / sum(sizes),
       rows = rows, e_src = fill(pos[src[keep]] - 1L, 0L),
       e_dst = fill(pos[dst[keep]] - 1L, 0L), e_g = fill(gr[keep], 0L),
       e_w = fill(w[keep], 0), edges = edges, pos = pos,
       edge_ids = split(seq_len(nrow(edges)), factor(e_comp, levels = seq_len(m))),
       n_trivial = sum(size == 1L), n_edges = length(keep))
}

# Exact sum over components and latent components of log |det(I - B)| from
# fitted couplings, by sparse LU. Also useful for fits made without the term.

#' Log Determinant of the GiVAE Graph Prior
#'
#' Computes the exact log-determinant term \eqn{\sum_l \log|\det(I - B_l)|} of
#' the simultaneous autoregressive prior of a fitted [GiVAE()] model, using the
#' fitted couplings. The residual terms of the ELBO do not contain it; adding
#' it (divided by the number of rows) gives the ELBO of the exact SAR prior.
#' Works for fits with and without `jacobian`, so the two can be compared.
#'
#' @param object A fitted [GiVAE()] model.
#' @return A list with `total`, `per_latent`, `per_row` (total divided by the
#'   number of rows), `elbo` (the ELBO per row without the term), `elbo_full`
#'   (with the term), `components` and `largest` (number and largest size of the
#'   non-trivial strongly connected components).
#' @export
givae_logdet <- function(object) {
  if (!inherits(object, "GiVAE")) stop("`object` must be a GiVAE fit.", call. = FALSE)
  neigh <- object$neighbors
  n <- object$n
  L <- object$latent_dim
  bl <- graph_blocks(neigh, n)
  out <- list(total = 0, per_latent = rep(0, L), per_row = 0, elbo = object$elbo,
              elbo_full = object$elbo, components = 0L, largest = 0L)
  if (bl$m == 0L) return(out)
  rho <- matrix(object$prior$rho, n, L)
  beta <- array(object$prior$beta, c(n, neigh$G, L))
  ed <- bl$edges
  per_latent <- numeric(L)
  for (l in seq_len(L)) {
    cf <- object$rho_max * rho[ed$src, l] * beta[cbind(ed$src, ed$g, l)] * ed$w
    for (s in seq_len(bl$m)) {
      e <- bl$edge_ids[[s]]
      r <- bl$sizes[s]
      B <- Matrix::sparseMatrix(i = bl$pos[ed$src[e]], j = bl$pos[ed$dst[e]],
                                x = cf[e], dims = c(r, r))
      d <- Matrix::determinant(Matrix::Diagonal(r) - B, logarithm = TRUE)
      per_latent[l] <- per_latent[l] + as.numeric(d$modulus)
    }
  }
  total <- sum(per_latent)
  list(total = total, per_latent = per_latent, per_row = total / n,
       elbo = object$elbo, elbo_full = object$elbo + total / n,
       components = bl$m, largest = bl$Smax)
}

# log |det(I - B_l)| summed over l for ONE block, differentiable in the
# couplings. rho: r x L, beta: r x G x L (tensors of the rows of the block);
# e_src, e_dst, e_g, e_w: padded edge vectors of the block (positions inside
# the block, 0-based; padded edges have weight 0). Returns a scalar tensor.
jacobian_block_logdet <- function(rho, beta, e_src, e_dst, e_g, e_w, r, L,
                                  rho_max, mode, terms, probes) {
  tf <- tensorflow::tf
  r <- as.integer(r)
  L <- as.integer(L)
  terms <- as.integer(terms)
  probes <- as.integer(probes)
  rho_e <- tf$gather(rho, e_src)                                   # E x L
  beta_e <- tf$gather(beta, e_src)                                 # E x G x L
  beta_e <- tf$gather(beta_e, e_g, axis = 1L, batch_dims = 1L)     # E x L
  scale <- tf$multiply(tf$constant(rho_max, dtype = "float32"),
                       tf$expand_dims(e_w, -1L))                   # E x 1
  c_e <- tf$multiply(scale, tf$multiply(rho_e, beta_e))            # E x L

  if (mode == "exact") {
    B <- tf$scatter_nd(indices = tf$stack(list(e_src, e_dst), axis = 1L),
                       updates = c_e,
                       shape = tf$constant(c(r, r, L), dtype = "int32"))   # r x r x L
    B <- tf$transpose(B, perm = c(2L, 0L, 1L))                     # L x r x r
    M <- tf$subtract(tf$eye(r, batch_shape = list(L)), B)
    sl <- tf$linalg$slogdet(M)
    logabs <- if (is.list(sl)) sl[[2]] else sl$log_abs_determinant
    return(tf$reduce_sum(logabs))
  }

  # stochastic: -sum_k tr(B^k) / k with Rademacher probes
  z <- tf$random$uniform(shape = c(r, L, probes), minval = 0L, maxval = 2L,
                         dtype = "int32")
  z <- tf$subtract(tf$multiply(tf$cast(z, "float32"), 2), 1)       # r x L x P
  c3 <- tf$expand_dims(c_e, -1L)                                   # E x L x 1
  v <- z
  acc <- tf$constant(0, dtype = "float32")
  for (k in seq_len(terms)) {
    v <- tf$math$unsorted_segment_sum(tf$multiply(tf$gather(v, e_dst), c3),
                                      e_src, num_segments = r)     # r x L x P
    tr_k <- tf$reduce_mean(tf$reduce_sum(tf$multiply(z, v), axis = 0L), axis = 1L)  # L
    acc <- tf$subtract(acc, tf$divide(tf$reduce_sum(tr_k),
                                      tf$constant(k, dtype = "float32")))
  }
  acc
}

# Returns a function without arguments that gives the Jacobian term of the
# loss as a scalar tensor: -(1 / N) * sum_s log det(I - B_s) / q_s for a
# sampled strongly connected component s (unbiased for the sum over all
# components). The loss is minus the ELBO, so the term always enters with this
# sign. `jac` is the list made in fit_vae_family(); `coef_fn(a)` maps
# auxiliary rows to list(rho = rows x L, beta = rows x G x L) with the networks
# of the model. The function is called inside the loss, i.e. during training.
jacobian_term_fn <- function(jac, coef_fn, L, rho_max) {
  function() {
    tf <- tensorflow::tf
    A_all <- tf$constant(jac$A, dtype = "float32")
    s <- if (jac$m > 1L) {
      logits <- tf$constant(matrix(log(jac$q), nrow = 1L), dtype = "float32")
      tf$squeeze(tf$random$categorical(logits, 1L, dtype = "int32"))
    } else {
      tf$constant(0L, dtype = "int32")
    }
    pick <- function(x, dtype) tf$gather(tf$constant(x, dtype = dtype), s)
    rows <- pick(jac$rows, "int32")
    cf <- coef_fn(tf$gather(A_all, rows))
    ld <- jacobian_block_logdet(
      cf$rho, cf$beta, pick(jac$e_src, "int32"), pick(jac$e_dst, "int32"),
      pick(jac$e_g, "int32"), pick(jac$e_w, "float32"), jac$Smax, L, rho_max,
      jac$mode, jac$terms, jac$probes)
    q_s <- tf$gather(tf$constant(jac$q, dtype = "float32"), s)
    tf$multiply(tf$constant(-1 / jac$N, dtype = "float32"), tf$divide(ld, q_s))
  }
}

# The ELBO loss, with the Jacobian term added if there is one.
loss_with_jacobian <- function(term) {
  if (is.null(term)) return(elbo_loss)
  function(y_true, y_pred) {
    tensorflow::tf$add(elbo_loss(y_true, y_pred), term())
  }
}