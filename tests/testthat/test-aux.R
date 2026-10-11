test_that("segmentation gives one indicator per segment", {
  A <- aux_matrix(aux_segments(25), matrix(0, 100, 0), time = 1:100)
  expect_equal(dim(A), c(100, 4))
  expect_true(all(rowSums(A) == 1))
  expect_equal(unname(colSums(A)), rep(25, 4))

  coords <- cbind(runif(200), runif(200))
  A <- aux_matrix(aux_segments(c(0.5, 0.5, 50), joint = c(1, 1, 2)),
                  matrix(0, 200, 0), coords = coords, time = rep(1:100, 2))
  expect_true(all(rowSums(A) == 2))
})

test_that("seasonal and weekly segments are added", {
  t <- as.Date("2024-01-01") + 0:59
  A <- aux_matrix(aux_segments(10, seasonal_period = 30, week_component = TRUE),
                  matrix(0, 60, 0), time = t)
  expect_true(all(rowSums(A) == 3))
})

test_that("new data are segmented with the training segments", {
  dat <- NonlinearBSS:::nlbss_data(matrix(0, 100, 0), time = 1:100,
                                   require_x = FALSE)
  spec <- NonlinearBSS:::fit_aux(aux_segments(25), dat)
  new <- NonlinearBSS:::nlbss_data(matrix(0, 3, 0), time = c(1, 60, 100),
                                   require_x = FALSE)
  A <- NonlinearBSS:::transform_aux(spec, new, NULL, FALSE)
  expect_equal(unname(A), unname(diag(4)[c(1, 3, 4), ]))
  far <- NonlinearBSS:::nlbss_data(matrix(0, 1, 0), time = 500, require_x = FALSE)
  expect_warning(A <- NonlinearBSS:::transform_aux(spec, far, NULL, FALSE),
                 "not present")
  expect_equal(sum(A), 0)
})

test_that("radial basis functions have the expected dimension and range", {
  coords <- cbind(runif(50), runif(50))
  A <- aux_matrix(aux_rbf(spatial_basis = c(2, 3), temporal_basis = c(4, 5)),
                  matrix(0, 50, 0), coords = coords, time = 1:50)
  expect_equal(ncol(A), 2^2 + 3^2 + 4 + 5)
  expect_true(all(A >= 0 & A <= 1))

  A <- aux_matrix(aux_rbf(spatial_basis = 3, kernel = "wendland"),
                  matrix(0, 50, 0), coords = coords)
  expect_equal(ncol(A), 9)

  # time series only: spatial basis functions are skipped
  A <- aux_matrix(aux_rbf(temporal_basis = c(3, 6), seasonal_period = 10),
                  matrix(0, 50, 0), time = 1:50)
  expect_equal(ncol(A), 3 + 6 + 6)
})

test_that("covariates are standardised and can be combined", {
  df <- data.frame(t = 1:20, elev = rnorm(20, 100, 10), y = rnorm(20))
  A <- aux_matrix(list(aux_covariates("elev"), aux_coords()), df, time = "t")
  expect_equal(ncol(A), 2)
  expect_equal(mean(A[, 1]), 0, tolerance = 1e-12)
  expect_equal(range(A[, 2]), c(0, 1))

  A <- aux_matrix(aux_rbf(temporal_basis = 3, elevation = "elev",
                          elevation_basis = 4), df, time = "t")
  expect_equal(ncol(A), 7)

  M <- matrix(rnorm(40), 20)
  expect_equal(aux_matrix(M, df, time = "t"), M, ignore_attr = TRUE)
})

test_that("auxiliary data errors are informative", {
  expect_error(aux_matrix(aux_rbf(), matrix(0, 5, 0)), "coordinates or time")
  expect_error(aux_matrix(aux_segments(c(1, 2)), matrix(0, 5, 0), time = 1:5),
               "length")
  expect_error(aux_matrix(aux_covariates("zz"), data.frame(a = 1:3)),
               "not found")
})

test_that("auxiliary specifications print", {
  expect_output(print(aux_rbf()), "radial basis")
  expect_output(print(aux_segments(c(1, 1, 5), joint = c(1, 1, 2))), "joint")
})
