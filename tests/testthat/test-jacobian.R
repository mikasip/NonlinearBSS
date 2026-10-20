skip_if_no_tf <- function() {
  skip_on_cran()
  skip_if_not(tf_available(), "TensorFlow not available")
}

# one block: exact and stochastic estimates agree with the sparse-LU value
test_that("block log determinants agree (exact, series, R reference)", {
  skip_if_no_tf()
  skip_if_not_installed("igraph")
  tf <- tensorflow::tf
  set.seed(2)
  n_s <- 6; L <- 2; rho_max <- 0.5
  path_adj <- lapply(seq_len(n_s), function(i) {
    nb <- c(i - 1L, i + 1L)
    nb[nb >= 1L & nb <= n_s]
  })
  g_s <- nlbss_graph(path_adj)
  ni <- NonlinearBSS:::build_neighbor_index(list(g_s), n_s)
  bl <- NonlinearBSS:::graph_blocks(ni, n_s)
  rho <- matrix(runif(n_s * L, 0.3, 0.9), n_s, L)
  beta <- array(1, c(n_s, 1, L))
  fake <- structure(list(neighbors = ni, n = n_s, latent_dim = L,
                         prior = list(rho = rho, beta = beta),
                         rho_max = rho_max, elbo = 0), class = c("GiVAE", "nlbss"))
  ref <- givae_logdet(fake)$total

  blk <- function(mode, terms = 1L, probes = 1L) {
    as.numeric(as.array(NonlinearBSS:::jacobian_block_logdet(
      tf$constant(rho, dtype = "float32"), tf$constant(beta, dtype = "float32"),
      tf$constant(bl$e_src[1, ], dtype = "int32"), tf$constant(bl$e_dst[1, ], dtype = "int32"),
      tf$constant(bl$e_g[1, ], dtype = "int32"), tf$constant(bl$e_w[1, ], dtype = "float32"),
      bl$Smax, L, rho_max, mode, terms, probes)))
  }
  expect_equal(blk("exact"), ref, tolerance = 1e-4)
  expect_equal(blk("stochastic", terms = 60L, probes = 4000L), ref, tolerance = 0.02)
})

test_that("GiVAE fits with the Jacobian term on and off", {
  skip_if_no_tf()
  skip_if_not_installed("igraph")
  set.seed(1)
  areal <- generate_areal_units(10, n_fine = 300, seed = 1)
  n_s <- nrow(areal$W); n_t <- 6
  src <- simulate_sources_areal(2, areal = areal, n_time = n_t, seed = 1)
  x <- mix_data(src$sources, 2)
  g_s <- nlbss_graph(areal$W)
  g_t <- nlbss_time_graph(n_t, each = n_s)
  for (jac in c("none", "exact", "stochastic")) {
    fit <- GiVAE(x, list(g_s, g_t), K = c(3, 3), latent_dim = 2, epochs = 1,
                 batch_size = 32, verbose = 0, seed = 1, jacobian = jac,
                 jacobian_terms = 10L)
    expect_equal(fit$jacobian$mode, jac)
    expect_false(anyNA(fit$S))
    ld <- givae_logdet(fit)
    expect_equal(ld$components, n_t)
    expect_true(is.finite(ld$elbo_full))
  }
  # acyclic: a directed time graph alone needs no term
  fit <- GiVAE(x, g_t, K = 3, latent_dim = 2, epochs = 1, batch_size = 32,
               verbose = 0, seed = 1, jacobian = "exact")
  expect_equal(fit$jacobian$mode, "none")
})