test_that("matrices and data frames are converted", {
  X <- matrix(rnorm(20), 10)
  d <- NonlinearBSS:::nlbss_data(X, time = 1:10)
  expect_equal(dim(d$X), c(10, 2))
  expect_equal(d$time, 1:10)
  expect_null(d$coords)

  df <- data.frame(t = 1:10, a = rnorm(10), b = rnorm(10), id = "s")
  d <- NonlinearBSS:::nlbss_data(df, time = "t", group = "id")
  expect_equal(colnames(d$X), c("a", "b"))
  expect_equal(d$group, rep("s", 10))
  out <- NonlinearBSS:::restore_format(d, d$X)
  expect_s3_class(out, "data.frame")
  expect_true(all(c("t", "id", "a", "b") %in% names(out)))
})

test_that("time series formats are converted and restored", {
  x <- ts(matrix(rnorm(30), 10), start = c(2020, 1), frequency = 12)
  d <- NonlinearBSS:::nlbss_data(x)
  expect_equal(d$time, as.numeric(time(x)))
  expect_s3_class(NonlinearBSS:::restore_format(d, d$X), "ts")

  skip_if_not_installed("xts")
  xx <- xts::xts(matrix(rnorm(20), 10), order.by = as.Date("2024-01-01") + 0:9)
  d <- NonlinearBSS:::nlbss_data(xx)
  expect_equal(d$weekday[1], 1L)
  out <- NonlinearBSS:::restore_format(d, d$X)
  expect_s3_class(out, "xts")
  expect_equal(zoo::index(out), zoo::index(xx))

  skip_if_not_installed("tsibble")
  tb <- tsibble::as_tsibble(
    data.frame(t = rep(1:5, 2), k = rep(c("a", "b"), each = 5), v = rnorm(10)),
    index = t, key = k)
  d <- NonlinearBSS:::nlbss_data(tb)
  expect_equal(d$group, rep(c("a", "b"), each = 5))
  expect_s3_class(NonlinearBSS:::restore_format(d, d$X), "tbl_ts")
})

test_that("spatial formats are converted and restored", {
  skip_if_not_installed("sf")
  P <- pollution_northern_italy[1:100, ]
  s <- sf::st_as_sf(P, coords = c("Longitude", "Latitude"), crs = 4326)
  d <- NonlinearBSS:::nlbss_data(s, time = "time",
                                 vars = c("mean_O3", "mean_NO2"))
  expect_equal(unname(d$coords[, 1]), P$Longitude)
  expect_equal(d$time, as.numeric(P$time))
  expect_s3_class(NonlinearBSS:::restore_format(d, d$X), "sf")

  skip_if_not_installed("sftime")
  st <- sftime::st_sftime(s, time_column_name = "time")
  d <- NonlinearBSS:::nlbss_data(st, vars = "mean_O3")
  expect_equal(d$time, as.numeric(P$time))
  expect_s3_class(NonlinearBSS:::restore_format(d, d$X), "sftime")
})

test_that("STFDF objects are converted and restored", {
  skip_if_not_installed("spacetime")
  skip_if_not_installed("sp")
  sp_pts <- sp::SpatialPoints(cbind(runif(4), runif(4)))
  tm <- as.POSIXct("2024-01-01", tz = "UTC") + 3600 * (1:5)
  stf <- spacetime::STFDF(sp_pts, tm, data.frame(a = rnorm(20), b = rnorm(20)))
  d <- NonlinearBSS:::nlbss_data(stf)
  expect_equal(nrow(d$coords), 20)
  expect_equal(length(unique(d$time)), 5)
  expect_s4_class(NonlinearBSS:::restore_format(d, d$X), "STFDF")
})

test_that("invalid input gives informative errors", {
  expect_error(NonlinearBSS:::nlbss_data(letters), "numeric")
  expect_error(NonlinearBSS:::nlbss_data(matrix(1:4, 2), time = 1:3), "one value")
  expect_error(NonlinearBSS:::nlbss_data(data.frame(a = 1:3), vars = "b"))
})
