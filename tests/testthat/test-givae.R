skip_if_no_tf <- function() {
  skip_on_cran()
  skip_if_not(tf_available(), "TensorFlow not available")
}

test_that("GiVAE fits with a spatial and a directed temporal graph", {
  skip_if_no_tf()
  set.seed(1)
  areal <- generate_areal_units(12, n_fine = 400, seed = 1)
  n_s <- nrow(areal$W)
  n_t <- 8
  src <- simulate_sources_areal(3, areal = areal, n_time = n_t, seed = 1)
  x <- mix_data(src$sources, 2)
  graphs <- list(nlbss_graph(areal$W, name = "space"),
                 nlbss_time_graph(n_t, each = n_s))

  fit <- GiVAE(x, graphs, K = c(4, 3), latent_dim = 3, epochs = 2,
               batch_size = 32, verbose = 0, seed = 1)
  expect_s3_class(fit, c("GiVAE", "iVAE", "nlbss"))
  expect_equal(dim(fit$S), dim(x))
  expect_equal(fit$neighbors$G, 2L)
  expect_true(is.finite(fit$elbo))
  expect_output(print(fit), "GiVAE")
  # coupling is component-specific: rho is n x L, beta is n x G x L
  expect_equal(dim(fit$prior$rho), c(nrow(x), 3))
  expect_equal(dim(fit$prior$beta), c(nrow(x), 2, 3))
  expect_equal(apply(fit$prior$beta, c(1, 3), sum), matrix(1, nrow(x), 3),
               tolerance = 1e-5)
  expect_true(all(predict(fit, type = "variance") > 0))
  expect_error(predict(fit, type = "forecast"), "not available")
  # trend is on by default
  expect_equal(dim(fit$prior$mean), c(nrow(x), 3))
  expect_equal(dim(predict(fit, type = "trend")), c(nrow(x), 3))
  fit0 <- GiVAE(x, graphs, K = c(4, 3), latent_dim = 3, epochs = 1,
                batch_size = 32, verbose = 0, seed = 1, trend = FALSE)
  expect_null(fit0$prior$mean)
  expect_error(predict(fit0, type = "trend"), "not available")

  # default auxiliary data are Laplacian eigenvectors, one block per graph
  expect_equal(fit$aux$specs[[1]]$n_features, 4 + 3)

  # a single graph over all rows, and missing values
  x_na <- x
  x_na[sample(length(x_na), 30)] <- NA
  fit2 <- GiVAE(x_na, nlbss_graph(areal$W), K = 3, latent_dim = 2, epochs = 1,
                batch_size = 32, verbose = 0, seed = 1)
  expect_true(fit2$spec$use_mask)
  expect_false(anyNA(fit2$S))
  expect_error(GiVAE(x[-1, ], graphs, latent_dim = 2, epochs = 1, verbose = 0),
               "multiple")
})