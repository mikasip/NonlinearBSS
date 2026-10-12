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
    covariates = covariate_values(spec, dat, values, training)
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
    covariates = function(x) (x - tf_row(spec$center)) / tf_row(spec$scale_by)
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
