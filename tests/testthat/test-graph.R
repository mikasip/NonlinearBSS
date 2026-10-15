path_adj <- function(n) {
  lapply(seq_len(n), function(i) {
    nb <- c(i - 1L, i + 1L)
    nb[nb >= 1L & nb <= n]
  })
}

n_s <- 4; n_t <- 3; n <- n_s * n_t
sp <- rep(seq_len(n_s), n_t)           # unit varies fastest within time
tm <- rep(seq_len(n_t), each = n_s)
g_s <- nlbss_graph(path_adj(n_s), name = "space")
g_t <- nlbss_time_graph(n_t, each = n_s)

test_that("graph layouts give node and context of each row", {
  lay <- NonlinearBSS:::graph_layout(g_s, n)
  expect_equal(lay$node, sp)
  expect_equal(lay$context, tm)
  lay <- NonlinearBSS:::graph_layout(g_t, n)
  expect_equal(lay$node, tm)
  expect_equal(lay$context, sp)
  # one node per observation
  g_n <- nlbss_graph(path_adj(n))
  expect_equal(NonlinearBSS:::graph_layout(g_n, n)$node, seq_len(n))
  expect_error(NonlinearBSS:::graph_layout(g_t, n + 1), "multiple")
})

test_that("neighbours are searched in the same context", {
  nb <- NonlinearBSS:::graph_neighbors(g_s, n)
  expect_true(all(mapply(function(i, v) all(tm[v] == tm[i]), seq_len(n), nb)))
  expect_equal(nb[[1]], 2L)
  expect_equal(sort(nb[[n_s + 2]]), n_s + c(1L, 3L))

  # directed time graph: only the previous time point of the same unit
  nbt <- NonlinearBSS:::graph_neighbors(g_t, n)
  expect_equal(lengths(nbt), ifelse(tm == 1, 0L, 1L))
  expect_equal(unlist(nbt[tm > 1]), seq_len(n - n_s))
  expect_true(all(sp[unlist(nbt)] == sp[tm > 1]))
  # undirected: also the next time point
  g_u <- nlbss_time_graph(n_t, each = n_s, directed = FALSE)
  expect_equal(lengths(NonlinearBSS:::graph_neighbors(g_u, n)),
               ifelse(tm == 2, 2L, 1L))
})

test_that("explicit layouts handle unsorted rows", {
  set.seed(1)
  o <- sample(n)
  g_o <- nlbss_graph(path_adj(n_s), node = sp[o], context = tm[o])
  nb_o <- NonlinearBSS:::graph_neighbors(g_o, n)
  nb <- NonlinearBSS:::graph_neighbors(g_s, n)
  expect_equal(lapply(seq_len(n), function(i) sort(o[nb_o[[i]]])),
               lapply(o, function(r) sort(nb[[r]])))
  expect_error(NonlinearBSS:::graph_layout(nlbss_graph(path_adj(n_s), node = sp), n),
               "same node and context")
  # missing cell: the neighbour does not exist, so it is dropped
  keep <- setdiff(seq_len(n), 2L)
  g_m <- nlbss_graph(path_adj(n_s), node = sp[keep], context = tm[keep])
  nb_m <- NonlinearBSS:::graph_neighbors(g_m, length(keep))
  expect_equal(length(nb_m[[1]]), 0L)
})

test_that("neighbour index and graph prior mean are consistent", {
  ni <- NonlinearBSS:::build_neighbor_index(list(g_s, g_t), n)
  expect_equal(ni$G, 2L)
  expect_equal(dim(ni$index), c(n, 2L * ni$M))
  expect_true(all(is.na(ni$index[tm == 1, ni$M + 1L])))

  ni1 <- NonlinearBSS:::build_neighbor_index(list(g_s), n)
  Z <- matrix(rnorm(2 * n), n)
  prior <- list(rho = matrix(1, n, 2), beta = array(1, c(n, 1, 2)))
  pm <- NonlinearBSS:::graph_prior_mean(prior, Z, ni1, rho_max = 0.5)
  nb <- NonlinearBSS:::graph_neighbors(g_s, n)
  expect_equal(pm, 0.5 * t(vapply(nb, function(v) colMeans(Z[v, , drop = FALSE]),
                                  numeric(2))))
  expect_error(nlbss_graph(list(integer(0), integer(0))), "no edges")
})

test_that("Laplacian eigenvectors skip one trivial vector per component", {
  r <- NonlinearBSS:::compute_laplacian_eigenvectors(path_adj(10), K = 3)
  expect_equal(dim(r$phi), c(10, 3))
  expect_true(all(r$lambda > 1e-8) && !is.unsorted(r$lambda))
  expect_equal(unname(colSums(r$phi)), rep(0, 3), tolerance = 1e-8)

  two <- c(path_adj(5), lapply(path_adj(5), function(v) v + 5L))
  r2 <- NonlinearBSS:::compute_laplacian_eigenvectors(two, K = 3)
  expect_equal(r2$n_components, 2L)
  expect_true(all(r2$lambda > 1e-8))

  iso <- list(2L, 1L, integer(0))     # isolated node counts as a component
  expect_equal(NonlinearBSS:::compute_laplacian_eigenvectors(iso, K = 1)$n_components, 2L)
})

test_that("Laplacian auxiliary data follow the graph layouts", {
  spec <- aux_laplacian(list(g_s, g_t), K = c(2, 1))
  A <- aux_matrix(spec, matrix(0, n, 0))
  expect_equal(dim(A), c(n, 3))
  # spatial block depends on the unit only, temporal block on the time only
  expect_equal(A[tm == 1, 1:2], A[tm == 2, 1:2], ignore_attr = TRUE)
  expect_equal(A[sp == 1, 3], A[sp == 2, 3], ignore_attr = TRUE)
  expect_equal(length(unique(A[, 3])), n_t)
  expect_output(print(spec), "Laplacian")
  expect_error(aux_laplacian(list(g_s, g_t), K = c(1, 2, 3)), "per graph")
})

test_that("sparse Laplacian eigensolver agrees with the analytic path-graph values", {
  skip_if_not_installed("RSpectra")
  n_big <- 800
  K <- 5
  r <- NonlinearBSS:::compute_laplacian_eigenvectors(path_adj(n_big), K = K,
                                                    dense_max = 100L)
  expect_equal(r$lambda, 2 - 2 * cos(pi * seq_len(K) / n_big), tolerance = 1e-6)
  rd <- NonlinearBSS:::compute_laplacian_eigenvectors(path_adj(n_big), K = K,
                                                     dense_max = Inf)
  expect_equal(abs(diag(cor(r$phi, rd$phi))), rep(1, K), tolerance = 1e-4)
})

test_that("a time graph with more than 500 nodes works with an explicit layout", {
  n_day <- 882; n_st <- 5; n_big <- n_day * n_st
  # rows ordered station within day, but given through an explicit layout
  day <- rep(seq_len(n_day), each = n_st)
  station <- rep(seq_len(n_st), n_day)
  g_day <- nlbss_time_graph(n_day, node = day, context = station)
  A <- aux_matrix(aux_laplacian(g_day, K = 5), matrix(0, n_big, 0))
  expect_equal(dim(A), c(n_big, 5))
  expect_equal(A[day == 10, 1], rep(A[day == 10, 1][1], n_st))   # constant within a day
  expect_equal(length(unique(A[, 1])), n_day)

  nb <- NonlinearBSS:::graph_neighbors(g_day, n_big)
  expect_equal(lengths(nb), ifelse(day == 1, 0L, 1L))
  expect_true(all(day[unlist(nb)] == day[day > 1] - 1L))
  expect_true(all(station[unlist(nb)] == station[day > 1]))
})

test_that("the temporal graph neighbours equal the iVAEar lag index", {
  n_s <- 4; n_t <- 6; n <- n_s * n_t
  time <- rep(seq_len(n_t), each = n_s)
  group <- rep(seq_len(n_s), n_t)
  g <- nlbss_time_graph(n_t, each = n_s)
  nb <- NonlinearBSS:::build_neighbor_index(list(g), n)$index[, 1]
  lag <- NonlinearBSS:::lag_index(time, as.character(group), 1)[, 1]
  expect_equal(as.integer(nb), as.integer(lag))
})

test_that("nlbss_dag keeps only earlier neighbours", {
  W <- matrix(0, 4, 4); W[cbind(1:3, 2:4)] <- 1; W <- pmax(W, t(W))
  g <- nlbss_dag(W, order = c(2, 1, 4, 3))      # node order: 2, 1, 4, 3
  expect_equal(g$adj_list, list(2L, integer(0), c(2L, 4L), integer(0)))
  pos <- rank(c(2, 1, 4, 3))
  expect_true(all(unlist(lapply(seq_along(g$adj_list), function(i)
    pos[g$adj_list[[i]]] < pos[i]))))
  # default order: neighbours with a smaller index
  expect_equal(nlbss_dag(W)$adj_list, list(integer(0), 1L, 2L, 3L))
})