# Model tests need TensorFlow; they are skipped on CRAN.

skip_if_no_tf <- function() {
  skip_on_cran()
  skip_if_not(tf_available(), "TensorFlow not available")
}

sim_st <- function(n_s = 20, n_t = 30, p = 3, ar = FALSE) {
  coords <- matrix(runif(2 * n_s), ncol = 2)[rep(seq_len(n_s), n_t), ]
  time <- rep(seq_len(n_t), each = n_s)
  src <- simulate_sources(p, coords = coords, time = time,
                          n_segments_space = 3, n_segments_time = 5,
                          ar_range = if (ar) c(-0.8, 0.8))
  list(x = mix_data(src$sources, 2), coords = coords, time = time,
       sources = src$sources)
}

test_that("iVAE fits with matrix, segment and rbf auxiliary data", {
  skip_if_no_tf()
  set.seed(1)
  d <- sim_st()
  fit <- iVAE(d$x, aux_segments(c(0.5, 0.5, 10)), coords = d$coords,
              time = d$time, epochs = 2, verbose = 0, seed = 1)
  expect_s3_class(fit, c("iVAE", "nlbss"))
  expect_equal(dim(fit$S), dim(d$x))
  expect_equal(unname(colMeans(fit$S)), rep(0, 3), tolerance = 1e-6)
  expect_true(is.finite(fit$elbo))
  expect_output(print(fit), "iVAE")
  expect_output(print(summary(fit)), "Prior variance")

  A <- aux_matrix(aux_coords(), d$x, coords = d$coords, time = d$time)
  fit2 <- iVAE(d$x, A, latent_dim = 2, epochs = 1, verbose = 0)
  expect_equal(ncol(fit2$S), 2)
  expect_error(predict(fit2, d$x[1:3, ], coords = d$coords[1:3, ],
                       time = d$time[1:3]), "aux")
  p2 <- predict(fit2, d$x[1:3, ], aux = A[1:3, ])
  expect_equal(unname(p2), unname(fit2$S[1:3, ]), tolerance = 1e-4)
})

test_that("predictions reproduce the training components", {
  skip_if_no_tf()
  set.seed(2)
  d <- sim_st()
  fit <- iVAE(d$x, aux_rbf(c(2, 3), c(3, 5)), coords = d$coords,
              time = d$time, epochs = 2, verbose = 0, seed = 1)
  p <- predict(fit, d$x[1:10, ], coords = d$coords[1:10, ], time = d$time[1:10])
  expect_equal(unname(p), unname(fit$S[1:10, ]), tolerance = 1e-4)
  tr <- predict(fit, type = "trend", coords = d$coords[1:4, ], time = 31:34)
  expect_equal(dim(tr), c(4, 3))
  v <- predict(fit, type = "variance")
  expect_true(all(v > 0))
  rec <- fitted(fit)
  expect_equal(dim(rec), dim(d$x))
  expect_equal(dim(decode(fit, fit$S[1:2, ])), c(2, 3))
  expect_error(predict(fit, type = "forecast"), "not available")
})

test_that("batch-wise auxiliary variables match precomputed ones", {
  skip_if_no_tf()
  d <- sim_st()
  df <- data.frame(d$x, s1 = d$coords[, 1], s2 = d$coords[, 2], t = d$time,
                   elev = rnorm(nrow(d$x), 100, 10))
  dat <- NonlinearBSS:::nlbss_data(df, coords = c("s1", "s2"), time = "t",
                                   aux_cols = "elev")
  spec <- list(
    aux_segments(c(0.5, 0.5, 4), joint = c(1, 1, 2), seasonal_period = 10,
                 week_component = TRUE),
    aux_segments(c(0.3, 0.3, 10)),
    aux_rbf(c(2, 3), c(3, 5), seasonal_period = 10, week_component = TRUE,
            kernel = "wendland", elevation = "elev", elevation_basis = 3),
    aux_coords(),
    aux_covariates("elev"),
    aux_covariates(matrix(rnorm(2 * nrow(d$x)), ncol = 2))
  )
  pb <- NonlinearBSS:::prepare_aux(spec, dat, batch = TRUE)
  pr <- NonlinearBSS:::prepare_aux(spec, dat, batch = FALSE)
  expect_lt(ncol(pb$raw), ncol(pr$raw))
  f <- NonlinearBSS:::aux_tf_function(pb$layer_config)
  Fb <- as.matrix(f(keras3::op_convert_to_tensor(pb$raw, dtype = "float32")))
  expect_equal(Fb, pr$raw, tolerance = 1e-5, ignore_attr = TRUE)

  fit <- iVAE(df, spec[1:5], coords = c("s1", "s2"), time = "t",
              vars = colnames(d$x), batch_aux = TRUE, epochs = 1, verbose = 0)
  expect_true(fit$aux$batch)
  expect_equal(nrow(predict(fit, df[1:3, ], type = "trend",
                            coords = c("s1", "s2"), time = "t")), 3)
})

test_that("validation data, callbacks and data-scale predictions work", {
  skip_if_no_tf()
  set.seed(7)
  d <- sim_st(ar = TRUE)
  tr <- d$time <= 25
  va <- list(x = d$x[!tr, ], coords = d$coords[!tr, ], time = d$time[!tr])
  state <- .Random.seed
  fit <- iVAEar(d$x[tr, ], aux_rbf(c(2, 3), c(3, 5)), coords = d$coords[tr, ],
                time = d$time[tr], validation_data = va, epochs = 30,
                verbose = 0, seed = 1,
                callbacks = list(keras3::callback_early_stopping(patience = 1)))
  expect_identical(.Random.seed, state)
  h <- fit$history
  expect_true(any(h$data == "validation"))
  expect_false(anyNA(h$value))
  expect_error(iVAE(d$x, aux_coords(), coords = d$coords, time = d$time,
                    validation_data = va, validation_split = 0.1),
               "not both")

  fx <- predict(fit, type = "forecast", coords = d$coords[1:20, ],
                time = rep(26, 20), space = "data")
  fc <- predict(fit, type = "forecast", coords = d$coords[1:20, ],
                time = rep(26, 20))
  expect_equal(fx, decode(fit, fc), ignore_attr = TRUE)
  expect_equal(colnames(fx), colnames(d$x))
  it <- predict(fit, type = "trend", coords = d$coords[1:4, ], time = 1:4,
                space = "data")
  expect_equal(dim(it), c(4, 3))
  expect_error(predict(fit, type = "variance", space = "data"), "available")
  expect_error(predict(fit, type = "forecast", coords = d$coords[1:2, ],
                       time = c(26, 26), space = "data", se = TRUE), "latent")
})

test_that("iVAEar fits, forecasts and handles unordered rows", {
  skip_if_no_tf()
  set.seed(3)
  d <- sim_st(ar = TRUE)
  o <- sample(nrow(d$x))
  fit <- iVAEar(d$x[o, ], aux_rbf(c(2, 3), c(3, 5)), ar_order = 2,
                coords = d$coords[o, ], time = d$time[o], epochs = 2,
                verbose = 0, seed = 1)
  expect_s3_class(fit, c("iVAEar", "iVAE", "nlbss"))
  expect_equal(ncol(fit$lags), 2)
  expect_equal(dim(predict(fit, type = "ar_coef")), c(nrow(d$x), 6))
  fc <- predict(fit, type = "forecast", coords = d$coords[1:40, ],
                time = rep(31:32, each = 20), se = TRUE)
  expect_equal(dim(fc$fit), c(40, 3))
  expect_true(all(fc$var > 0))
  # one step forecast variance equals the prior variance
  v <- predict(fit, type = "variance", coords = d$coords[1:20, ],
               time = rep(31, 20))
  expect_equal(unname(fc$var[1:20, ]), unname(v), tolerance = 1e-5)
  expect_error(iVAEar(d$x, aux_coords(), coords = d$coords, epochs = 1,
                      verbose = 0), "time")
})

test_that("TCL and VAE fit time series and return the input format", {
  skip_if_no_tf()
  set.seed(4)
  src <- simulate_sources(2, time = 1:400, n_segments_time = 8,
                          mean_range = c(0, 0))
  x <- ts(mix_data(src$sources, 2))
  tc <- TCL(x, n_segments = 8, epochs = 2, verbose = 0, seed = 1)
  expect_s3_class(components(tc), "ts")
  expect_equal(length(unique(tc$labels)), 8)
  expect_equal(unname(predict(tc, x[1:5, ])), unname(tc$S[1:5, ]),
               tolerance = 1e-4)
  expect_error(fitted(tc), "decoder")
  expect_equal(components(tc, format = "matrix"), tc$S)
  expect_equal(generics::components(tc), components(tc))
  if (requireNamespace("ICS", quietly = TRUE)) {
    expect_equal(ICS::components(tc), components(tc))
  }

  v <- VAE(x, epochs = 2, verbose = 0, seed = 1)
  expect_s3_class(fitted(v), "ts")
  expect_error(predict(v, type = "trend"), "not available")
})

test_that("missing values and count data are handled", {
  skip_if_no_tf()
  set.seed(5)
  d <- sim_st()
  x <- d$x
  x[sample(length(x), 50)] <- NA
  fit <- iVAE(x, aux_coords(), coords = d$coords, time = d$time, epochs = 1,
              verbose = 0)
  expect_true(fit$spec$use_mask)
  expect_false(anyNA(fit$S))

  counts <- matrix(rpois(length(d$x), exp(d$x / 3)), nrow(d$x))
  fit <- iVAE(counts, aux_coords(), coords = d$coords, time = d$time,
              error_dist = "poisson", error_dist_sigma = c(0.05, 0.01),
              epochs = 1, verbose = 0)
  expect_true(all(fitted(fit) > 0))
  expect_error(iVAE(-counts, aux_coords(), coords = d$coords, time = d$time,
                    error_dist = "poisson", epochs = 1, verbose = 0),
               "non-negative")
})

test_that("models are saved and loaded", {
  skip_if_no_tf()
  d <- sim_st()
  fit <- iVAEar(d$x, aux_coords(), coords = d$coords, time = d$time,
                epochs = 1, verbose = 0)
  file <- tempfile(fileext = ".rds")
  on.exit(unlink(file))
  save_nlbss(fit, file)
  fit2 <- load_nlbss(file)
  args <- list(d$x[1:5, ], coords = d$coords[1:5, ], time = d$time[1:5])
  expect_equal(do.call(predict, c(list(fit2), args)),
               do.call(predict, c(list(fit), args)))
  # plain saveRDS works, the models are rebuilt on first use
  saveRDS(fit, file)
  fit3 <- readRDS(file)
  expect_equal(decode(fit3, fit$S[1:2, ]), decode(fit, fit$S[1:2, ]))
})

test_that("spatial input formats give outputs in the same format", {
  skip_if_no_tf()
  skip_if_not_installed("sf")
  P <- pollution_northern_italy[pollution_northern_italy$time <= as.Date("2020-01-20"), ]
  s <- sf::st_as_sf(P, coords = c("Longitude", "Latitude"), crs = 4326)
  fit <- iVAEar(s, aux_rbf(2, 3), time = "time",
                vars = c("mean_O3", "mean_NO2", "mean_PM10"),
                epochs = 1, verbose = 0)
  ic <- components(fit)
  expect_s3_class(ic, "sf")
  expect_equal(nrow(ic), nrow(P))
  pdf(NULL)
  on.exit(grDevices::dev.off())
  expect_silent(plot(fit))
  expect_silent(plot(fit, "series", series = 2))
  expect_silent(plot(fit, "history"))
})

test_that("mshap gives scaled mean absolute SHAP values", {
  skip_if_no_tf()
  set.seed(6)
  d <- sim_st()
  fit <- iVAE(d$x, aux_coords(), coords = d$coords, time = d$time,
              epochs = 1, verbose = 0)
  ms <- mshap(fit, n_sample = 50, n_background = 10, seed = 1)
  expect_s3_class(ms, "mshap")
  expect_equal(dim(ms$mshap), c(3, 3))
  expect_equal(unname(rowSums(ms$scaled)), rep(1, 3))
  expect_equal(unname(colMeans(ms$scaled)), sort(colMeans(ms$scaled), decreasing = TRUE),
               ignore_attr = TRUE)
  expect_equal(length(ms$shap), 3)
  expect_equal(dim(ms$shap[[1]]), c(50, 3))
  # SHAP values add up to the prediction minus the baseline
  pred <- decode(fit, ms$X)
  expect_equal(unname(rowSums(ms$shap[[1]]) + ms$kernelshap$baseline[1]),
               unname(pred[, 1]), tolerance = 1e-3)
  expect_output(print(ms), "Average")
  pdf(NULL)
  on.exit(grDevices::dev.off())
  expect_silent(plot(ms))
  expect_error(mshap(TCL(d$x, epochs = 1, verbose = 0)), "iVAE")
})

test_that("the number of latent components can be selected by pAIC", {
  skip_if_no_tf()
  set.seed(8)
  src <- simulate_sources(2, time = 1:400, n_segments_time = 8)
  x <- mix_data(src$sources, n_layers = 2, output_dim = 4)
  sel <- select_latent_dim(x, latent_dims = c(3, 1, 2), aux = aux_segments(50),
                           time = 1:400, epochs = 2, verbose = 0, seed = 1)
  expect_s3_class(sel, "latent_dim_selection")
  expect_equal(sel$table$latent_dim, 1:3)
  expect_equal(sel$table$pAIC, sel$table$elbo - 1:3)
  expect_equal(sel$selected, sel$table$latent_dim[which.max(sel$table$pAIC)])
  expect_equal(vapply(sel$models, function(m) m$latent_dim, numeric(1)), 1:3)
  expect_equal(pAIC(sel$models[[2]]), sel$models[[2]]$elbo - 2)
  expect_equal(pAIC(sel$models), sel$table$pAIC)
  expect_output(print(sel), "Selected")
  pdf(NULL)
  on.exit(grDevices::dev.off())
  expect_silent(plot(sel))
  expect_silent(plot(sel, which = "pAIC"))

  other <- iVAE(x, aux_coords(), latent_dim = 2, time = 1:400, epochs = 1,
                verbose = 0)
  expect_warning(latent_dim_selection(list(sel$models[[1]], other)),
                 "comparable")
  expect_null(latent_dim_selection(sel$models, keep_models = FALSE)$models)
  expect_error(select_latent_dim(x, 1:2, method = TCL), "method")
})
