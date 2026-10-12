# Auxiliary data specifications --------------------------------------------
#
# An auxiliary data specification describes how auxiliary variables are
# formed from the spatial coordinates, time points and other columns of the
# data. It is fitted once to the training data (storing e.g. the range of the
# coordinates) and can then be applied to new data for prediction.

new_aux <- function(type, ...) {
  structure(list(type = type, ..., fitted = FALSE),
            class = c(paste0("aux_", type), "nlbss_aux"))
}

#' Auxiliary Data Based on Segmentation
#'
#' Specifies auxiliary variables that indicate the spatial, temporal or
#' spatio-temporal segment each observation belongs to.
#'
#' The segmentation is done in the domain formed by the spatial coordinates
#' and the time points (in this order, time being the last dimension). Each
#' dimension is divided into intervals of length `size`. Dimensions with the
#' same value in `joint` are segmented jointly, so that a segment is a
#' (hyper)rectangle in those dimensions. Each group of jointly segmented
#' dimensions gives one block of indicator variables. For example, for
#' spatio-temporal data with two spatial dimensions, `joint = c(1, 1, 1)`
#' gives a full spatio-temporal segmentation, `joint = c(1, 1, 2)` segments
#' space and time separately and `joint = c(1, 2, 3)` segments every dimension
#' separately. Segmentations were used as auxiliary data for spatial data by
#' \insertCite{sipila2024spatial;textual}{NonlinearBSS} and for
#' spatio-temporal data by \insertCite{sipila2024modelling;textual}{NonlinearBSS}.
#'
#' For a seasonal period, the time dimension is segmented within the season
#' (time modulo `seasonal_period`) and an additional indicator block gives the
#' season number. `week_component = TRUE` adds indicators for the day of the
#' week.
#'
#' @param size Segment sizes, a single value or one value per dimension
#'   (spatial dimensions first, then time). For time series this is the
#'   segment length in time units.
#' @param joint An integer vector, one value per dimension, defining which
#'   dimensions are segmented jointly. Default is to segment all dimensions
#'   jointly.
#' @param seasonal_period Optional length of a seasonal period in time units.
#' @param max_season Optional maximum season number. Later seasons are
#'   treated as the last season, which is useful for forecasting beyond the
#'   training period.
#' @param week_component Logical. Add day of the week indicators? Meaningful
#'   for daily data only.
#' @return An object of class `aux_segments` to be passed as `aux` to
#'   [iVAE()], [iVAEar()] or [TCL()].
#' @seealso [aux_rbf()], [aux_coords()], [aux_covariates()], [aux_matrix()]
#' @references \insertAllCited{}
#' @examples
#' # time series: segments of length 50
#' aux <- aux_segments(50)
#' head(aux_matrix(aux, matrix(rnorm(300), 100), time = 1:100))
#'
#' # spatio-temporal data: space and time segmented separately
#' aux_segments(c(0.2, 0.2, 10), joint = c(1, 1, 2))
#' @export
aux_segments <- function(size, joint = NULL, seasonal_period = NULL,
                         max_season = NULL, week_component = FALSE) {
  stopifnot(is.numeric(size), all(size > 0))
  new_aux("segments", size = size, joint = joint,
          seasonal_period = seasonal_period, max_season = max_season,
          week_component = isTRUE(week_component))
}

#' Auxiliary Data Based on Radial Basis Functions
#'
#' Specifies auxiliary variables formed by spatial and temporal radial basis
#' functions \insertCite{sipila2024spatial,sipila2024modelling}{NonlinearBSS}.
#'
#' The spatial coordinates are min-max normalised to \eqn{[0, 1]}. For a
#' resolution level \eqn{H} in `spatial_basis`, a regular grid of \eqn{H}
#' node points per dimension is placed with spacing \eqn{1/(H+2)} from the
#' borders and the basis functions are
#' \eqn{v(\|s - o_i\| / \zeta_H)}{v(||s - o_i|| / zeta_H)} with
#' \eqn{\zeta_H = 2.5 / H}, where \eqn{v} is the Gaussian kernel
#' \eqn{v(d) = \exp(-d^2)}{v(d) = exp(-d^2)} or the Wendland kernel
#' \insertCite{wendlandkernel1995}{NonlinearBSS}. Similar basis functions are
#' used in DeepKriging \insertCite{chen2020deepkriging}{NonlinearBSS}.
#'
#' For a temporal resolution level \eqn{G} in `temporal_basis`, \eqn{G}
#' evenly spaced node points are placed inside the observed time range and
#' Gaussian basis functions with standard deviation equal to the node
#' spacing are used. Several resolution levels capture both large scale and
#' fine scale variation.
#'
#' If `seasonal_period` is given, the temporal basis functions are formed for
#' the time within the season and indicator variables are added for the
#' season number. `week_component = TRUE` adds day of the week indicators.
#' An additional variable, typically elevation, can be included with its own
#' one-dimensional basis functions.
#'
#' The spatial basis functions are used if the data have spatial coordinates,
#' the temporal ones if the data have time points. The specification works
#' therefore for spatial data, time series and spatio-temporal data.
#'
#' @param spatial_basis Integer vector of spatial resolution levels.
#' @param temporal_basis Integer vector of temporal resolution levels.
#' @param kernel Spatial kernel, `"gaussian"` or `"wendland"`.
#' @param seasonal_period Optional length of a seasonal period in time units.
#' @param max_season Optional maximum season number, see [aux_segments()].
#' @param week_component Logical. Add day of the week indicators?
#' @param elevation Optional name of a column of the data (or a numeric
#'   vector with one value per observation) giving an additional variable such
#'   as elevation. Use a column name if predictions for new data are needed.
#' @param elevation_basis Integer vector of resolution levels for
#'   `elevation`.
#' @return An object of class `aux_rbf`.
#' @seealso [aux_segments()], [aux_coords()], [aux_covariates()],
#'   [aux_matrix()]
#' @references \insertAllCited{}
#' @examples
#' coords <- matrix(runif(200), ncol = 2)
#' A <- aux_matrix(aux_rbf(spatial_basis = c(2, 5)), matrix(0, 100, 0),
#'                 coords = coords)
#' dim(A)
#' @export
aux_rbf <- function(spatial_basis = c(2, 9), temporal_basis = c(9, 17, 37),
                    kernel = c("gaussian", "wendland"), seasonal_period = NULL,
                    max_season = NULL, week_component = FALSE,
                    elevation = NULL, elevation_basis = c(5, 10)) {
  new_aux("rbf", spatial_basis = as.integer(spatial_basis),
          temporal_basis = as.integer(temporal_basis),
          kernel = match.arg(kernel), seasonal_period = seasonal_period,
          max_season = max_season, week_component = isTRUE(week_component),
          elevation = elevation, elevation_basis = as.integer(elevation_basis))
}

#' Auxiliary Data Based on Coordinates
#'
#' Specifies the min-max normalised spatial coordinates and time points
#' themselves as auxiliary variables.
#'
#' @return An object of class `aux_coords`.
#' @seealso [aux_segments()], [aux_rbf()], [aux_covariates()]
#' @examples
#' aux_matrix(aux_coords(), matrix(0, 5, 0), time = 1:5)
#' @export
aux_coords <- function() new_aux("coords")

#' Auxiliary Data Given by Covariates
#'
#' Specifies observed covariates as auxiliary variables, for example
#' satellite-derived predictors in addition to spatio-temporal radial basis
#' functions \insertCite{sipila2026missing}{NonlinearBSS}.
#'
#' @param x Either names of columns in the data, or a numeric matrix, data
#'   frame or vector with one row per observation. If columns are named, the
#'   values for new data are taken from the new data in [predict()][predict.nlbss].
#' @param scale Logical. Standardise the covariates to zero mean and unit
#'   variance?
#' @return An object of class `aux_covariates`.
#' @details A plain matrix given as `aux` to the fitting functions is
#'   interpreted as `aux_covariates(x, scale = FALSE)`.
#' @seealso [aux_segments()], [aux_rbf()], [aux_coords()]
#' @references \insertAllCited{}
#' @examples
#' aux_covariates(matrix(rnorm(20), 10))
#' @export
aux_covariates <- function(x, scale = TRUE) {
  if (!is.character(x)) {
    x <- as.matrix(x)
    storage.mode(x) <- "double"
  }
  new_aux("covariates", values = x, scale = isTRUE(scale))
}

#' Auxiliary Data Based on Moran Eigenvector Maps
#'
#' Specifies auxiliary variables from Moran's eigenvector maps (MEM), built
#' from a symmetric spatial neighborhood matrix or adjacency list. This is
#' intended for areal data where the auxiliary variation is driven by the
#' connectivity graph rather than by coordinates.
#'
#' @param neighborhood A symmetric numeric matrix with one row and column per
#'   areal unit. The matrix is interpreted as the adjacency/weight matrix and
#'   is symmetrised internally if needed; do not row-standardise it.
#' @param adj_list Optional list of neighbor indices with one element per areal
#'   unit. Each entry should contain 1-based indices of adjacent units.
#' @param K Number of MEM eigenvectors to retain.
#' @param positive_only Keep only positive-eigenvalue MEMs? This is the usual
#'   choice for spatial areal data where positive spatial autocorrelation is
#'   expected.
#' @return An object of class `aux_moran`.
#' @seealso [aux_rbf()], [aux_segments()], [aux_covariates()]
#' @examples
#' n <- 10
#' W <- matrix(0, n, n)
#' W[cbind(1:n, c(2:n, 1))] <- 1
#' W[cbind(1:n, c(n, 1:(n - 1)))] <- 1
#' aux_moran(neighborhood = W, K = 3)
#' @export
aux_moran <- function(neighborhood = NULL, adj_list = NULL, K = 10L,
                      positive_only = TRUE) {
  if (is.null(neighborhood) && is.null(adj_list)) {
    stop("Either `neighborhood` or `adj_list` must be supplied.", call. = FALSE)
  }
  if (!is.null(neighborhood) && !is.null(adj_list)) {
    stop("Provide only one of `neighborhood` or `adj_list`.", call. = FALSE)
  }
  if (!is.null(neighborhood)) {
    neighborhood <- as.matrix(neighborhood)
    storage.mode(neighborhood) <- "double"
  }
  new_aux("moran", neighborhood = neighborhood, adj_list = adj_list,
          K = as.integer(K), positive_only = isTRUE(positive_only))
}

#' @export
print.nlbss_aux <- function(x, ...) {
  cat(format(x), sep = "\n")
  invisible(x)
}

#' @export
format.nlbss_aux <- function(x, ...) {
  desc <- switch(x$type,
    segments = paste0("segmentation, size = ", paste(x$size, collapse = ", "),
                      if (!is.null(x$joint)) paste0(", joint = ",
                                                    paste(x$joint, collapse = ", "))),
    rbf = paste0("radial basis functions, spatial = ",
                 paste(x$spatial_basis, collapse = ", "), ", temporal = ",
                 paste(x$temporal_basis, collapse = ", "), " (", x$kernel, ")"),
    coords = "normalised coordinates",
    covariates = paste0("covariates", if (is.character(x$values))
      paste0(" (", paste(x$values, collapse = ", "), ")") else
        paste0(" (", ncol(x$values), " columns)")),
    moran = paste0("Moran eigenvector maps (K = ", x$K,")")
  )
  extra <- c(
    if (!is.null(x$seasonal_period)) paste0("seasonal period ", x$seasonal_period),
    if (isTRUE(x$week_component)) "day of week",
    if (!is.null(x$elevation)) "elevation"
  )
  if (length(extra)) desc <- paste0(desc, " + ", paste(extra, collapse = " + "))
  if (isTRUE(x$fitted)) desc <- paste0(desc, " [", x$n_features, " variables]")
  paste("Auxiliary data:", desc)
}

#' Compute an Auxiliary Data Matrix
#'
#' Computes the auxiliary variables defined by one or more auxiliary data
#' specifications for given data. The fitting functions do this
#' automatically; the function is useful for inspecting the auxiliary data or
#' for using it with other methods.
#'
#' @param aux An auxiliary data specification such as [aux_rbf()], a numeric
#'   matrix, or a list of these.
#' @param x The data, in any format accepted by [iVAE()]. Only the
#'   coordinates, time points and columns referred to by `aux` are used.
#' @inheritParams iVAE
#' @return A numeric matrix with one row per observation. The fitted
#'   specifications are stored in the attribute `"aux"`.
#' @examples
#' A <- aux_matrix(aux_segments(25), matrix(0, 100, 0), time = 1:100)
#' colSums(A)
#' @export
aux_matrix <- function(aux, x, coords = NULL, time = NULL, group = NULL) {
  dat <- nlbss_data(x, coords = coords, time = time, group = group,
                    vars = character(0), aux_cols = aux_columns(aux),
                    require_x = FALSE)
  prep <- prepare_aux(aux, dat, batch = FALSE)
  structure(prep$raw, aux = prep$specs)
}

# Internal machinery -------------------------------------------------------

as_aux_list <- function(aux) {
  if (inherits(aux, "nlbss_aux")) return(list(aux))
  if (is.matrix(aux) || is.data.frame(aux) || (is.numeric(aux) && is.null(dim(aux)))) {
    return(list(aux_covariates(aux, scale = FALSE)))
  }
  if (is.list(aux)) return(unlist(lapply(aux, as_aux_list), recursive = FALSE))
  stop("`aux` must be an auxiliary data specification, a numeric matrix ",
       "or a list of these.", call. = FALSE)
}

aux_columns <- function(aux) {
  if (is.null(aux)) return(character(0))
  unlist(lapply(as_aux_list(aux), function(s) {
    c(if (is.character(s$values)) s$values,
      if (is.character(s$elevation)) s$elevation)
  }))
}

default_aux <- function(dat) {
  if (is.null(dat$coords) && is.null(dat$time)) {
    stop("`aux` is needed: the data have no spatial coordinates or time ",
         "points from which auxiliary data could be formed.", call. = FALSE)
  }
  aux_rbf()
}

# Fit all specifications and compute the raw auxiliary input matrix. In batch
# mode the radial basis functions are not computed here; instead the raw
# coordinates and time points are passed to the network.
prepare_aux <- function(aux, dat, batch = FALSE) {
  specs <- lapply(as_aux_list(aux), function(s) {
    if (inherits(s, "aux_moran")) {
      fit_aux.aux_moran(s, dat)
    } else {
      fit_aux(s, dat)
    }
  })
  aux_inputs(list(specs = specs, batch = batch), dat, values = NULL,
             training = TRUE)
}

transform_aux_spec <- function(spec, dat, values, training) {
  if (inherits(spec, "aux_moran")) {
    transform_aux.aux_moran(spec, dat, values, training)
  } else {
    transform_aux(spec, dat, values, training)
  }
}

aux_inputs <- function(prep, dat, values = NULL, training = FALSE) {
  specs <- prep$specs
  value_specs <- which(vapply(specs, has_values, logical(1)))
  if (!training && length(value_specs) > 0 && is.null(values)) {
    stop("The auxiliary data contain values given as vectors or matrices. ",
         "Provide them for the new data with the argument `aux`.",
         call. = FALSE)
  }
  if (!is.null(values) && !is.list(values)) values <- list(values)
  if (!is.null(values) && length(values) != length(value_specs)) {
    stop("`aux` must give values for ", length(value_specs),
         " auxiliary specification(s).", call. = FALSE)
  }
  get_vals <- function(i) {
    if (training || !has_values(specs[[i]])) return(NULL)
    values[[match(i, value_specs)]]
  }
  batch <- isTRUE(prep$batch)
  blocks <- lapply(seq_along(specs), function(i) {
    if (batch) {
      as.matrix(aux_raw_block(specs[[i]], dat, get_vals(i), training))
    } else {
      as.matrix(transform_aux_spec(specs[[i]], dat, get_vals(i), training))
    }
  })
  raw <- do.call(cbind, blocks)
  storage.mode(raw) <- "double"
  config <- NULL
  if (batch) {
    ends <- cumsum(vapply(blocks, ncol, numeric(1)))
    config <- lapply(seq_along(specs), function(i) {
      list(spec = specs[[i]], cols = seq(ends[i] - ncol(blocks[[i]]) + 1, ends[i]))
    })
  }
  list(specs = specs, batch = batch, raw = raw, layer_config = config,
       n_raw = ncol(raw),
       n_features = sum(vapply(specs, function(s) s$n_features, numeric(1))))
}

has_values <- function(spec) {
  (spec$type == "covariates" && !is.character(spec$values)) ||
    (spec$type == "rbf" && is.numeric(spec$elevation))
}

fit_aux <- function(spec, dat) UseMethod("fit_aux")
transform_aux <- function(spec, dat, values, training) UseMethod("transform_aux")

seg_locations <- function(dat) {
  L <- cbind(dat$coords, time = dat$time)
  if (is.null(L) || ncol(L) == 0) {
    stop("The data have no spatial coordinates or time points to segment.",
         call. = FALSE)
  }
  L
}

season_split <- function(t, period, max_season = NULL) {
  season <- floor(t / period)
  if (!is.null(max_season)) season <- pmin(season, max_season)
  list(season = season, within = t - period * floor(t / period) + 1)
}

one_hot <- function(x, levels) {
  M <- matrix(0, length(x), length(levels))
  idx <- match(x, levels)
  ok <- !is.na(idx)
  M[cbind(which(ok), idx[ok])] <- 1
  M
}

#' @export
fit_aux.aux_segments <- function(spec, dat) {
  L <- seg_locations(dat)
  d <- ncol(L)
  if (length(spec$size) == 1) spec$size <- rep(spec$size, d)
  if (length(spec$size) != d) {
    stop("`size` must have length 1 or ", d, " (spatial dimensions and time).",
         call. = FALSE)
  }
  if (is.null(spec$joint)) spec$joint <- rep(1L, d)
  if (length(spec$joint) != d) stop("`joint` must have length ", d, ".", call. = FALSE)
  if ((!is.null(spec$seasonal_period) || spec$week_component) && is.null(dat$time)) {
    stop("Seasonal and weekly components need time points.", call. = FALSE)
  }
  if (!is.null(spec$seasonal_period)) {
    ss <- season_split(dat$time, spec$seasonal_period, spec$max_season)
    L[, d] <- ss$within
    spec$seasons <- seq(min(ss$season), max(ss$season))
  }
  spec$dims <- d
  spec$mins <- apply(L, 2, min)
  spec$levels <- lapply(unique(spec$joint), function(g) {
    unique(segment_keys(L, spec, which(spec$joint == g)))
  })
  # segment cell indices of the levels, used for batch-wise computation
  spec$level_cells <- lapply(spec$levels, function(lv) {
    matrix(as.numeric(unlist(strsplit(lv, "_", fixed = TRUE))),
           nrow = length(lv), byrow = TRUE)
  })
  spec$n_features <- sum(lengths(spec$levels)) + length(spec$seasons) +
    7 * spec$week_component
  spec$fitted <- TRUE
  spec
}

segment_keys <- function(L, spec, dims) {
  cells <- lapply(dims, function(j) floor((L[, j] - spec$mins[j]) / spec$size[j]))
  do.call(paste, c(cells, sep = "_"))
}

#' @export
transform_aux.aux_segments <- function(spec, dat, values, training) {
  L <- seg_locations(dat)
  if (ncol(L) != spec$dims) {
    stop("The new data have a different number of dimensions than the ",
         "training data.", call. = FALSE)
  }
  blocks <- list()
  if (!is.null(spec$seasonal_period)) {
    ss <- season_split(dat$time, spec$seasonal_period, max(spec$seasons))
    L[, spec$dims] <- ss$within
  }
  groups <- unique(spec$joint)
  unseen <- FALSE
  for (k in seq_along(groups)) {
    keys <- segment_keys(L, spec, which(spec$joint == groups[k]))
    unseen <- unseen || any(!keys %in% spec$levels[[k]])
    blocks[[k]] <- one_hot(keys, spec$levels[[k]])
  }
  if (unseen) {
    warning("Some observations fall in segments not present in the training ",
            "data; their segment indicators are set to zero.", call. = FALSE)
  }
  if (!is.null(spec$seasonal_period)) {
    blocks <- c(blocks, list(one_hot(pmax(ss$season, min(spec$seasons)), spec$seasons)))
  }
  if (spec$week_component) blocks <- c(blocks, list(one_hot(dat$weekday, 0:6)))
  do.call(cbind, blocks)
}

# Integer segment labels (for TCL), combining all segmentation groups.
segment_labels <- function(spec, dat) {
  A <- transform_aux(spec, dat, NULL, TRUE)
  as.integer(factor(apply(A, 1, paste, collapse = "")))
}

#' @export
fit_aux.aux_coords <- function(spec, dat) {
  L <- seg_locations(dat)
  spec$mins <- apply(L, 2, min)
  spec$ranges <- safe_range(apply(L, 2, max) - spec$mins)
  spec$n_features <- ncol(L)
  spec$fitted <- TRUE
  spec
}

#' @export
transform_aux.aux_coords <- function(spec, dat, values, training) {
  L <- seg_locations(dat)
  sweep(sweep(L, 2, spec$mins, "-"), 2, spec$ranges, "/")
}

covariate_values <- function(spec, dat, values, training) {
  v <- spec$values
  if (is.character(v)) {
    cols <- dat$template$columns
    if (is.null(cols) || !all(v %in% names(cols))) {
      stop("Auxiliary covariate column(s) not found in the data: ",
           paste(setdiff(v, names(cols)), collapse = ", "), call. = FALSE)
    }
    v <- as.matrix(cols[v])
  } else if (!training) {
    v <- as.matrix(values)
  }
  storage.mode(v) <- "double"
  if (nrow(v) != nrow(dat$X)) {
    stop("Auxiliary covariates must have one row per observation.", call. = FALSE)
  }
  if (anyNA(v)) stop("Auxiliary covariates must not contain missing values.", call. = FALSE)
  v
}

#' @export
fit_aux.aux_covariates <- function(spec, dat) {
  v <- covariate_values(spec, dat, NULL, TRUE)
  spec$center <- if (spec$scale) colMeans(v) else rep(0, ncol(v))
  spec$scale_by <- if (spec$scale) safe_range(apply(v, 2, sd)) else rep(1, ncol(v))
  spec$n_features <- ncol(v)
  spec$fitted <- TRUE
  spec
}

#' @export
transform_aux.aux_covariates <- function(spec, dat, values, training) {
  v <- covariate_values(spec, dat, values, training)
  if (ncol(v) != spec$n_features) {
    stop("Auxiliary covariates must have ", spec$n_features, " columns.",
         call. = FALSE)
  }
  sweep(sweep(v, 2, spec$center, "-"), 2, spec$scale_by, "/")
}

safe_range <- function(r) {
  r[!is.finite(r) | r == 0] <- 1
  r
}

adjacency_matrix <- function(adj_list, n = NULL) {
  if (is.null(adj_list)) {
    stop("`adj_list` must not be NULL.", call. = FALSE)
  }
  if (is.matrix(adj_list) || is.data.frame(adj_list)) {
    W <- as.matrix(adj_list)
    storage.mode(W) <- "double"
    if (nrow(W) != ncol(W)) {
      stop("The adjacency matrix must be square.", call. = FALSE)
    }
    return((W + t(W)) / 2)
  }
  if (!is.list(adj_list)) {
    stop("`adj_list` must be a list or a square matrix.", call. = FALSE)
  }
  if (is.null(n)) n <- length(adj_list)
  rows <- integer(0)
  cols <- integer(0)
  for (i in seq_len(n)) {
    nbrs <- as.integer(adj_list[[i]])
    if (length(nbrs) == 0L) next
    if (any(nbrs < 1L | nbrs > n)) {
      stop("Adjacency indices must be between 1 and the number of areal units.",
           call. = FALSE)
    }
    rows <- c(rows, rep.int(i, length(nbrs)))
    cols <- c(cols, nbrs)
  }
  W <- Matrix::sparseMatrix(i = rows, j = cols, x = 1.0, dims = c(n, n))
  W <- (W + Matrix::t(W)) / 2
  as.matrix(W)
}

compute_MEM_vectors <- function(adj_list, K = 10L, positive_only = TRUE) {
  W <- adjacency_matrix(adj_list)
  if (nrow(W) != ncol(W)) {
    stop("The adjacency matrix must be square.", call. = FALSE)
  }
  if (!isSymmetric(W)) {
    W <- (W + t(W)) / 2
  }
  W_tilde <- W - outer(rowMeans(W), rowMeans(W), "+") + mean(W)
  eig <- eigen(W_tilde, symmetric = TRUE)
  if (isTRUE(positive_only)) {
    pos_idx <- which(eig$values > 1e-8)
    if (length(pos_idx) == 0L) {
      stop("No MEM vectors with positive eigenvalues were found.", call. = FALSE)
    }
    K <- min(as.integer(K), length(pos_idx))
    sel_idx <- pos_idx[seq_len(K)]
  } else {
    ord <- order(abs(eig$values), decreasing = TRUE)
    K <- min(as.integer(K), max(1L, nrow(W) - 1L))
    sel_idx <- ord[seq_len(K)]
  }
  mem <- eig$vectors[, sel_idx, drop = FALSE]
  for (k in seq_len(ncol(mem))) {
    idx <- which.max(abs(mem[, k]))
    if (mem[idx, k] < 0) mem[, k] <- -mem[, k]
  }
  colnames(mem) <- paste0("MEM_", seq_len(ncol(mem)))
  values <- eig$values[sel_idx]
  list(
    mem = mem,
    values = values,
    moran_i = values / sum(W),
    W_tilde = W_tilde
  )
}

#' @export
fit_aux.aux_moran <- function(spec, dat) {
  if (is.null(spec$neighborhood) && is.null(spec$adj_list)) {
    stop("The Moran auxiliary data are not fully specified.", call. = FALSE)
  }
  W <- if (!is.null(spec$neighborhood)) spec$neighborhood else {
    adjacency_matrix(spec$adj_list, n = nrow(dat$X))
  }
  n_areal <- nrow(W)
  n_obs <- nrow(dat$X)
  if (n_obs != n_areal && (n_obs %% n_areal) != 0L) {
    stop("The Moran neighborhood matrix must have one row and column per areal unit, or a repeated areal-by-time layout.",
         call. = FALSE)
  }
  mem <- compute_MEM_vectors(W, K = spec$K, positive_only = spec$positive_only)
  spec$mem <- mem$mem
  spec$values <- mem$values
  spec$moran_i <- mem$moran_i
  spec$W_tilde <- mem$W_tilde
  spec$n_features <- ncol(mem$mem)
  spec$fitted <- TRUE
  spec
}

#' @export
transform_aux.aux_moran <- function(spec, dat, values, training) {
  if (is.null(spec$mem)) {
    stop("The Moran auxiliary variables have not been fitted yet.", call. = FALSE)
  }
  n_areal <- nrow(spec$mem)
  n_obs <- nrow(dat$X)
  if (n_obs == n_areal) {
    return(spec$mem)
  }
  if (n_obs %% n_areal == 0L) {
    reps <- n_obs / n_areal
    return(spec$mem[rep(seq_len(n_areal), times = reps), , drop = FALSE])
  }
  stop("The new data must contain the same areal units as the fitted Moran auxiliary data.",
       call. = FALSE)
}

# Radial basis functions ---------------------------------------------------

elevation_values <- function(spec, dat, values, training) {
  if (is.null(spec$elevation)) return(NULL)
  e <- spec$elevation
  if (is.character(e)) {
    cols <- dat$template$columns
    if (is.null(cols) || !e %in% names(cols)) {
      stop("Elevation column '", e, "' not found in the data.", call. = FALSE)
    }
    e <- cols[[e]]
  } else if (!training) {
    e <- values
  }
  e <- as.numeric(e)
  if (length(e) != nrow(dat$X) || anyNA(e)) {
    stop("Elevation must have one non-missing value per observation.",
         call. = FALSE)
  }
  e
}

#' @export
fit_aux.aux_rbf <- function(spec, dat) {
  spec$use_space <- !is.null(dat$coords) && length(spec$spatial_basis) > 0
  spec$use_time <- !is.null(dat$time) && length(spec$temporal_basis) > 0
  if (!spec$use_space && !spec$use_time) {
    stop("Radial basis functions need spatial coordinates or time points.",
         call. = FALSE)
  }
  if ((!is.null(spec$seasonal_period) || spec$week_component) && is.null(dat$time)) {
    stop("Seasonal and weekly components need time points.", call. = FALSE)
  }
  n_feat <- 0
  if (spec$use_space) {
    spec$coord_min <- apply(dat$coords, 2, min)
    spec$coord_range <- safe_range(apply(dat$coords, 2, max) - spec$coord_min)
    spec$spatial_dim <- ncol(dat$coords)
    n_feat <- n_feat + sum(spec$spatial_basis^spec$spatial_dim)
  }
  if (spec$week_component) n_feat <- n_feat + 7
  if (!is.null(spec$seasonal_period)) {
    ss <- season_split(dat$time, spec$seasonal_period, spec$max_season)
    spec$seasons <- seq(min(ss$season), max(ss$season))
    spec$time_min <- 1
    spec$time_max <- spec$seasonal_period
    n_feat <- n_feat + length(spec$seasons)
  } else if (!is.null(dat$time)) {
    spec$time_min <- min(dat$time)
    spec$time_max <- max(dat$time)
  }
  if (spec$use_time) n_feat <- n_feat + sum(spec$temporal_basis)
  e <- elevation_values(spec, dat, NULL, TRUE)
  if (!is.null(e)) {
    spec$elevation_min <- min(e)
    spec$elevation_max <- max(e)
    n_feat <- n_feat + sum(spec$elevation_basis)
  }
  spec$n_features <- n_feat
  spec$fitted <- TRUE
  spec
}

#' @export
transform_aux.aux_rbf <- function(spec, dat, values, training) {
  check_rbf_data(spec, dat)
  rbf_features(spec, dat$coords, dat$time, dat$weekday,
               elevation_values(spec, dat, values, training))
}

check_rbf_data <- function(spec, dat) {
  if (spec$use_space && (is.null(dat$coords) || ncol(dat$coords) != spec$spatial_dim)) {
    stop("The new data must have ", spec$spatial_dim, " spatial coordinates.",
         call. = FALSE)
  }
  if ((spec$use_time || !is.null(spec$seasonal_period) || spec$week_component) &&
      is.null(dat$time)) {
    stop("The new data must have time points.", call. = FALSE)
  }
}

rbf_knots <- function(H, d) {
  k1 <- seq(1 / (H + 2), 1 - 1 / (H + 2), length.out = H)
  as.matrix(expand.grid(replicate(d, k1, simplify = FALSE)))
}

temporal_knots <- function(G, tmin, tmax) {
  k <- seq(tmin, tmax, length.out = G + 2)[-c(1, G + 2)]
  kappa <- if (G > 1) k[2] - k[1] else (tmax - tmin) / 2
  list(knots = k, kappa = if (kappa > 0) kappa else 1)
}

wendland <- function(d) ifelse(d < 1, (1 - d)^6 * (35 * d^2 + 18 * d + 3) / 3, 0)

rbf_features <- function(spec, coords, time, weekday, elevation) {
  feats <- list()
  if (spec$use_space) {
    s <- sweep(sweep(coords, 2, spec$coord_min, "-"), 2, spec$coord_range, "/")
    for (H in spec$spatial_basis) {
      K <- rbf_knots(H, spec$spatial_dim)
      d2 <- outer(rowSums(s^2), rowSums(K^2), "+") - 2 * s %*% t(K)
      d <- sqrt(pmax(d2, 0)) / (2.5 / H)
      feats[[length(feats) + 1]] <- if (spec$kernel == "gaussian") exp(-d^2) else wendland(d)
    }
  }
  if (spec$week_component) feats[[length(feats) + 1]] <- one_hot(weekday, 0:6)
  tt <- time
  if (!is.null(spec$seasonal_period)) {
    ss <- season_split(time, spec$seasonal_period, max(spec$seasons))
    feats[[length(feats) + 1]] <- one_hot(pmax(ss$season, min(spec$seasons)), spec$seasons)
    tt <- ss$within
  }
  if (spec$use_time) {
    for (G in spec$temporal_basis) {
      tk <- temporal_knots(G, spec$time_min, spec$time_max)
      feats[[length(feats) + 1]] <- exp(-0.5 * (outer(tt, tk$knots, "-") / tk$kappa)^2)
    }
  }
  if (!is.null(elevation)) {
    for (E in spec$elevation_basis) {
      ek <- temporal_knots(E, spec$elevation_min, spec$elevation_max)
      feats[[length(feats) + 1]] <- exp(-0.5 * (outer(elevation, ek$knots, "-") / ek$kappa)^2)
    }
  }
  M <- do.call(cbind, feats)
  storage.mode(M) <- "double"
  M
}
