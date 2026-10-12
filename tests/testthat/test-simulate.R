test_that("mix_data keeps dimensions and is injective in shape", {
  z <- matrix(rnorm(300), ncol = 3)
  expect_equal(dim(mix_data(z)), c(100, 3))
  expect_equal(dim(mix_data(z, n_layers = 3, nonlinearity = "xtanh",
                            output_dim = 5)), c(100, 5))
  expect_true(all(mix_data(z, 2, final_activation = "exp") > 0))
  expect_error(mix_data(z, output_dim = 2), "at least")
  # one layer is linear
  x <- mix_data(z, 1)
  expect_equal(qr(cbind(1, z))$rank, qr(cbind(1, z, x))$rank)
})

test_that("simulate_sources gives segment-wise sources", {
  src <- simulate_sources(2, time = 1:500, n_segments_time = 5)
  expect_equal(dim(src$sources), c(500, 2))
  expect_equal(length(unique(src$labels)), 5)

  coords <- matrix(runif(40), ncol = 2)[rep(1:20, 30), ]
  src <- simulate_sources(3, coords = coords, time = rep(1:30, each = 20),
                          n_segments_space = 3, n_segments_time = 4,
                          ar_range = c(0.5, 0.9))
  expect_equal(dim(src$sources), c(600, 3))
  expect_false(anyNA(src$sources))
  expect_error(simulate_sources(2, coords = coords, ar_range = c(0, 1)), "time")
})

test_that("absolute_mean_correlation finds the best permutation", {
  z <- matrix(rnorm(600), ncol = 3)
  expect_equal(absolute_mean_correlation(-z[, c(3, 1, 2)], z), 1)
  expect_equal(absolute_mean_correlation(diag(3)), 1)
  expect_lt(absolute_mean_correlation(matrix(rnorm(600), ncol = 3), z), 0.3)
  expect_equal(absolute_mean_correlation(cor(z[, 1:2], z)), 1)
})

test_that("lag_index finds previous observations of each series", {
  key <- rep(c("a", "b"), each = 4)
  time <- c(1, 2, 3, 4, 1, 2, 4, 5)
  lags <- NonlinearBSS:::lag_index(time, key, 2)
  expect_equal(lags[, 1], c(NA, 1, 2, 3, NA, 5, NA, 7))
  expect_equal(lags[, 2], c(NA, NA, 1, 2, NA, NA, 6, NA))
  expect_error(NonlinearBSS:::lag_index(c(1, 1), c("a", "a"), 1), "one observation")
})

test_that("generate_areal_units and simulate_sources_areal produce real areal data", {
  areal <- generate_areal_units(6, n_fine = 200, seed_type = "uniform", seed = 1)
  expect_equal(dim(areal$W), c(6, 6))
  expect_equal(length(unique(areal$area_id)), 6)

  src <- simulate_sources_areal(3, areal = areal, n_time = 10,
                                n_segments_time = 3, seed = 2)
  expect_equal(dim(src$sources), c(60, 3))
  expect_equal(dim(src$W), c(6, 6))
  expect_false(anyNA(src$sources))
})
