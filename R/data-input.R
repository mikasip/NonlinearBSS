# Conversion between the supported input formats and the internal
# representation used by all methods.
#
# An "nlbss_data" object is a list with
#   X        numeric n x p matrix of observed variables (may have 0 columns)
#   coords   numeric n x d matrix of spatial coordinates or NULL
#   time     numeric vector of time points or NULL
#   time_raw time points in their original class (Date, POSIXct, ...) or NULL
#   weekday  integer day of the week (0 = Sunday) or NULL
#   group    character series identifier or NULL
#   format   the input format, used to restore outputs
#   template format specific information needed to restore outputs

nlbss_data <- function(x, coords = NULL, time = NULL, group = NULL,
                       vars = NULL, aux_cols = character(0),
                       require_x = TRUE) {
  dat <- as_nlbss_data(x, coords = coords, time = time, group = group,
                       vars = vars, aux_cols = unique(aux_cols))
  validate_nlbss_data(dat, require_x = require_x)
}

as_nlbss_data <- function(x, ...) UseMethod("as_nlbss_data")

#' @export
as_nlbss_data.default <- function(x, coords = NULL, time = NULL, group = NULL,
                                  vars = NULL, aux_cols = character(0), ...) {
  if (is.data.frame(x)) {
    return(df_to_nlbss(x, coords, time, group, vars, format = "data.frame",
                       aux_cols = aux_cols))
  }
  if (is.numeric(x) && is.null(dim(x))) x <- matrix(x, ncol = 1)
  if (!is.matrix(x) || !(is.numeric(x) || is.logical(x) || all(is.na(x)))) {
    stop("`x` must be a numeric matrix, a data frame or one of the ",
         "supported time series or spatial classes.", call. = FALSE)
  }
  x <- select_vars(x, vars)
  new_nlbss_data(x, coords = coords, time_raw = time, group = group,
                 format = "matrix", template = list(rownames = rownames(x)))
}

#' @export
as_nlbss_data.ts <- function(x, coords = NULL, time = NULL, group = NULL,
                             vars = NULL, aux_cols = character(0), ...) {
  X <- matrix(unclass(x), ncol = NCOL(x))
  colnames(X) <- if (is.null(colnames(x))) "Series 1" else colnames(x)
  X <- select_vars(X, vars)
  if (is.null(time)) time <- as.numeric(stats::time(x))
  new_nlbss_data(X, coords = coords, time_raw = time, group = group,
                 format = "ts", template = list(tsp = stats::tsp(x)))
}

#' @export
as_nlbss_data.zoo <- function(x, coords = NULL, time = NULL, group = NULL,
                              vars = NULL, aux_cols = character(0), ...) {
  X <- as.matrix(zoo::coredata(x))
  if (is.null(colnames(X))) colnames(X) <- paste0("X", seq_len(ncol(X)))
  X <- select_vars(X, vars)
  index <- zoo::index(x)
  if (is.null(time)) time <- index
  format <- if (inherits(x, "xts")) "xts" else "zoo"
  new_nlbss_data(X, coords = coords, time_raw = time, group = group,
                 format = format, template = list(index = index))
}

#' @export
as_nlbss_data.tbl_ts <- function(x, coords = NULL, time = NULL, group = NULL,
                                 vars = NULL, aux_cols = character(0), ...) {
  require_pkg("tsibble")
  index <- tsibble::index_var(x)
  keys <- tsibble::key_vars(x)
  if (is.null(time)) time <- index
  if (is.null(group) && length(keys) > 0) group <- keys
  df <- as.data.frame(x)
  if (is.null(vars)) {
    vars <- tsibble::measured_vars(x)
    vars <- vars[vapply(df[vars], is.numeric, logical(1))]
  }
  dat <- df_to_nlbss(df, coords, time, group, vars, format = "tsibble",
                     aux_cols = aux_cols)
  dat$template$index <- index
  dat$template$keys <- keys
  dat
}

#' @export
as_nlbss_data.sf <- function(x, coords = NULL, time = NULL, group = NULL,
                             vars = NULL, aux_cols = character(0), ...) {
  require_pkg("sf")
  geometry <- sf::st_geometry(x)
  df <- sf::st_drop_geometry(x)
  if (is.null(coords)) coords <- sf_coordinates(geometry)
  format <- "sf"
  if (inherits(x, "sftime")) {
    format <- "sftime"
    time_column <- attr(x, "time_column")
    if (is.null(time)) time <- time_column
  }
  dat <- df_to_nlbss(as.data.frame(df), coords, time, group, vars,
                     format = format, aux_cols = aux_cols)
  dat$template$geometry <- geometry
  if (format == "sftime") dat$template$time_column <- time_column
  dat
}

#' @export
as_nlbss_data.STFDF <- function(x, coords = NULL, time = NULL, group = NULL,
                                vars = NULL, aux_cols = character(0), ...) {
  require_pkg("sp")
  require_pkg("spacetime")
  n_s <- length(x@sp)
  n_t <- nrow(x@time)
  if (is.null(coords)) {
    coords <- sp::coordinates(x@sp)[rep(seq_len(n_s), n_t), , drop = FALSE]
  }
  if (is.null(time)) time <- rep(zoo::index(x@time), each = n_s)
  df <- as.data.frame(x@data)
  dat <- df_to_nlbss(df, coords, time, group, vars, format = "STFDF",
                     aux_cols = aux_cols)
  dat$template$sp <- x@sp
  dat$template$time <- x@time
  dat$template$endTime <- x@endTime
  dat
}

# Data frames: coords, time, group and vars may be column names.
df_to_nlbss <- function(df, coords, time, group, vars, format,
                        aux_cols = character(0)) {
  missing_cols <- setdiff(aux_cols, names(df))
  if (length(missing_cols) > 0) {
    stop("Column(s) used for auxiliary data not found: ",
         paste(missing_cols, collapse = ", "), call. = FALSE)
  }
  used <- character(0)
  take <- function(spec) {
    if (is.character(spec) && all(spec %in% names(df))) {
      used <<- c(used, spec)
      return(df[spec])
    }
    spec
  }
  coords <- take(coords)
  time <- take(time)
  group <- take(group)
  if (is.data.frame(time)) time <- time[[1]]
  if (is.data.frame(group)) {
    group <- if (ncol(group) == 1) group[[1]] else interaction(group, drop = TRUE)
  }
  if (is.null(vars)) {
    vars <- setdiff(names(df)[vapply(df, is.numeric, logical(1))],
                    c(used, aux_cols))
  }
  bad <- vars[!vapply(df[vars], function(v) is.numeric(v) || is.logical(v),
                      logical(1))]
  if (length(bad) > 0) {
    stop("Variables must be numeric: ", paste(bad, collapse = ", "),
         call. = FALSE)
  }
  X <- as.matrix(df[vars])
  storage.mode(X) <- "double"
  new_nlbss_data(X, coords = coords, time_raw = time, group = group,
                 format = format,
                 template = list(id = if (length(used)) df[unique(used)] else NULL,
                                 columns = if (length(aux_cols)) df[aux_cols] else NULL,
                                 rownames = attr(df, "row.names")))
}

new_nlbss_data <- function(X, coords, time_raw, group, format, template) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (is.null(colnames(X)) && ncol(X) > 0) colnames(X) <- paste0("X", seq_len(ncol(X)))
  if (!is.null(coords)) {
    coords <- as.matrix(coords)
    storage.mode(coords) <- "double"
    if (is.null(colnames(coords))) colnames(coords) <- paste0("s", seq_len(ncol(coords)))
  }
  if (!is.null(group)) group <- as.character(group)
  time <- if (is.null(time_raw)) NULL else time_to_numeric(time_raw)
  structure(list(
    X = X, coords = coords, time = time, time_raw = time_raw,
    weekday = if (is.null(time_raw)) NULL else weekday_of(time_raw, time),
    group = group, format = format, template = template
  ), class = "nlbss_data")
}

validate_nlbss_data <- function(dat, require_x = TRUE) {
  n <- nrow(dat$X)
  if (require_x && ncol(dat$X) == 0) {
    stop("No numeric variables found in `x`.", call. = FALSE)
  }
  if (!is.null(dat$coords)) {
    if (nrow(dat$coords) != n) stop("`coords` must have one row per observation.", call. = FALSE)
    if (anyNA(dat$coords)) stop("`coords` must not contain missing values.", call. = FALSE)
  }
  if (!is.null(dat$time)) {
    if (length(dat$time) != n) stop("`time` must have one value per observation.", call. = FALSE)
    if (anyNA(dat$time)) stop("`time` must not contain missing values.", call. = FALSE)
  }
  if (!is.null(dat$group) && length(dat$group) != n) {
    stop("`group` must have one value per observation.", call. = FALSE)
  }
  dat
}

time_to_numeric <- function(t) {
  if (inherits(t, "POSIXt")) return(as.numeric(as.POSIXct(t)) / 86400)
  if (inherits(t, "Date")) return(as.numeric(t))
  out <- suppressWarnings(as.numeric(t))
  if (length(t) > 0 && all(is.na(out))) {
    stop("Time points must be numeric, Date or POSIXct.", call. = FALSE)
  }
  out
}

weekday_of <- function(t_raw, t_num) {
  if (inherits(t_raw, c("Date", "POSIXt"))) return(as.POSIXlt(t_raw)$wday)
  as.integer(floor(t_num) %% 7)
}

sf_coordinates <- function(geometry) {
  types <- unique(as.character(sf::st_geometry_type(geometry)))
  if (!all(types == "POINT")) {
    geometry <- suppressWarnings(sf::st_centroid(geometry))
  }
  xy <- sf::st_coordinates(geometry)
  xy[, intersect(colnames(xy), c("X", "Y", "Z")), drop = FALSE]
}

# Identifier of the series (location or group) each observation belongs to.
series_key <- function(dat) {
  if (!is.null(dat$group)) return(dat$group)
  if (!is.null(dat$coords)) {
    return(do.call(paste, c(lapply(seq_len(ncol(dat$coords)), function(j) {
      sprintf("%.12g", dat$coords[, j])
    }), sep = "_")))
  }
  rep("1", nrow(dat$X))
}

# Restore a matrix with one row per observation to the input format.
restore_format <- function(dat, M, format = c("input", "matrix")) {
  format <- match.arg(format)
  M <- as.matrix(M)
  if (format == "matrix" || dat$format == "matrix") {
    rownames(M) <- dat$template$rownames
    return(M)
  }
  df <- as.data.frame(M, optional = TRUE)
  names(df) <- colnames(M)
  id <- dat$template$id
  switch(dat$format,
    data.frame = if (is.null(id)) df else cbind(id, df),
    ts = stats::ts(M, start = dat$template$tsp[1], frequency = dat$template$tsp[3]),
    xts = {
      require_pkg("xts")
      xts::xts(M, order.by = dat$template$index)
    },
    zoo = zoo::zoo(M, order.by = dat$template$index),
    tsibble = {
      require_pkg("tsibble")
      tsibble::as_tsibble(cbind(id, df), index = dat$template$index,
                          key = dplyr_all_of(dat$template$keys))
    },
    sf = {
      require_pkg("sf")
      out <- if (is.null(id)) df else cbind(id, df)
      sf::st_sf(out, geometry = dat$template$geometry)
    },
    sftime = {
      require_pkg("sftime")
      out <- if (is.null(id)) df else cbind(id, df)
      out <- sf::st_sf(out, geometry = dat$template$geometry)
      sftime::st_as_sftime(out, time_column_name = dat$template$time_column)
    },
    STFDF = spacetime::STFDF(dat$template$sp, dat$template$time, df,
                             endTime = dat$template$endTime),
    M
  )
}

# tsibble's `key` argument uses tidy selection; a character vector works
# directly as long as it is not empty.
dplyr_all_of <- function(keys) if (length(keys) == 0) NULL else keys

require_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package '", pkg, "' is needed for this data format. ",
         "Please install it.", call. = FALSE)
  }
}

select_vars <- function(X, vars) {
  if (is.null(vars)) return(X)
  if (length(vars) == 0) return(X[, 0, drop = FALSE])
  if (is.character(vars) && !all(vars %in% colnames(X))) {
    stop("Variables not found: ", paste(setdiff(vars, colnames(X)), collapse = ", "),
         call. = FALSE)
  }
  X[, vars, drop = FALSE]
}
