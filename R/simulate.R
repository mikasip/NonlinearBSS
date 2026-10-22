#' Simulate Nonstationary Latent Sources
#'
#' Simulates independent latent components whose mean and variance, and
#' optionally autoregressive coefficient, change over spatial and/or
#' temporal segments. Such sources are identifiable by [iVAE()], [iVAEar()]
#' and [TCL()].
#'
#' @details
#' Spatial segments are formed by choosing `n_segments_space` of the distinct
#' locations at random as centres and assigning every location to the
#' nearest centre. Temporal segments are formed in the same way from the
#' distinct time points. For spatio-temporal data each combination of a
#' spatial and a temporal segment is its own segment.
#'
#' In each segment, each component has its own mean drawn from
#' \eqn{U}(`mean_range`) and standard deviation drawn from
#' \eqn{U}(`sd_range`). If `ar_range` is given, each component of each series
#' (location) follows a nonstationary AR(1) process
#' \deqn{z(t) = \mu(t) + \gamma(t) (z(t - 1) - \mu(t - 1)) + \sigma(t) \epsilon(t)}{
#' z(t) = mu(t) + gamma(t) (z(t-1) - mu(t-1)) + sigma(t) eps(t)}
#' with segment-wise coefficients \eqn{\gamma} drawn from \eqn{U}(`ar_range`)
#' and standard normal \eqn{\epsilon}; otherwise the observations are
#' independent normal.
#'
#' @param p Number of components.
#' @param coords Optional matrix of spatial coordinates, one row per
#'   observation.
#' @param time Optional vector of time points, one per observation. At least
#'   one of `coords` and `time` is needed.
#' @param n_segments_space Number of spatial segments.
#' @param n_segments_time Number of temporal segments.
#' @param mean_range Range of the segment means.
#' @param sd_range Range of the segment standard deviations.
#' @param ar_range Optional range of the autoregressive coefficients.
#'   Requires `time`.
#' @return A list with elements `sources` (an n x p matrix), `labels`
#'   (segment of each observation), `coords` and `time`.
#' @seealso [mix_data()]
#' @examples
#' # time series with variance changing in 10 segments
#' src <- simulate_sources(2, time = 1:1000, n_segments_time = 10)
#' plot.ts(src$sources)
#'
#' # spatial data
#' coords <- matrix(runif(1000), ncol = 2)
#' src <- simulate_sources(3, coords = coords, n_segments_space = 5)
#'
#' # spatio-temporal data with autoregressive sources
#' coords <- matrix(runif(40), ncol = 2)[rep(1:20, 50), ]
#' src <- simulate_sources(3, coords = coords, time = rep(1:50, each = 20),
#'                         n_segments_space = 3, n_segments_time = 5,
#'                         ar_range = c(-0.9, 0.9))
#' @export
simulate_sources <- function(p, coords = NULL, time = NULL,
                             n_segments_space = 10, n_segments_time = 10,
                             mean_range = c(-5, 5), sd_range = c(0.1, 5),
                             ar_range = NULL) {
  if (is.null(coords) && is.null(time)) {
    stop("Give `coords`, `time` or both.", call. = FALSE)
  }
  if (!is.null(ar_range) && is.null(time)) {
    stop("Autoregressive sources need `time`.", call. = FALSE)
  }
  if (!is.null(coords)) coords <- as.matrix(coords)
  n <- if (!is.null(coords)) nrow(coords) else length(time)
  lab_space <- rep(1L, n)
  if (!is.null(coords)) {
    key <- do.call(paste, c(as.data.frame(coords), sep = "_"))
    first <- !duplicated(key)
    U <- coords[first, , drop = FALSE]
    centres <- U[sample(nrow(U), min(n_segments_space, nrow(U))), , drop = FALSE]
    d2 <- outer(rowSums(U^2), rowSums(centres^2), "+") - 2 * U %*% t(centres)
    lab_space <- max.col(-d2, ties.method = "first")[match(key, key[first])]
  }
  lab_time <- rep(1L, n)
  if (!is.null(time)) {
    ut <- sort(unique(time))
    centres <- sample(ut, min(n_segments_time, length(ut)))
    lab_time <- max.col(-abs(outer(ut, centres, "-")),
                        ties.method = "first")[match(time, ut)]
  }
  labels <- as.integer(factor(paste(lab_space, lab_time)))
  k <- max(labels)
  draw <- function(range) matrix(runif(k * p, range[1], range[2]), k, p)
  mu <- draw(mean_range)[labels, , drop = FALSE]
  sigma <- draw(sd_range)[labels, , drop = FALSE]
  eps <- matrix(rnorm(n * p), n, p)
  if (is.null(ar_range)) {
    z <- mu + sigma * eps
  } else {
    gamma <- draw(ar_range)[labels, , drop = FALSE]
    series <- if (is.null(coords)) rep("1", n) else
      do.call(paste, c(as.data.frame(coords), sep = "_"))
    lag <- lag_index(time, series, 1)[, 1]
    z <- matrix(NA_real_, n, p)
    rank <- match(time, sort(unique(time)))
    for (r in sort(unique(rank))) {
      i <- which(rank == r)
      j <- lag[i]
      ok <- !is.na(j)
      zi <- mu[i, , drop = FALSE] + sigma[i, , drop = FALSE] * eps[i, , drop = FALSE]
      zi[ok, ] <- zi[ok, ] + gamma[i[ok], , drop = FALSE] *
        (z[j[ok], , drop = FALSE] - mu[j[ok], , drop = FALSE])
      z[i, ] <- zi
    }
  }
  colnames(z) <- paste0("Z", seq_len(p))
  list(sources = z, labels = labels, coords = coords, time = time)
}

#' Nonlinear Mixing of Data
#'
#' Mixes data with a random injective mixing function formed by a
#' multilayer perceptron. Useful for simulation studies.
#'
#' @details
#' The mixing function \eqn{f_L} with \eqn{L} layers is
#' \deqn{f_1(z) = B_1 z,\quad f_l(z) = B_l\, \omega(f_{l-1}(z)),\ l = 2, \dots, L,}{
#' f_1(z) = B_1 z,  f_l(z) = B_l omega(f_(l-1)(z)),}
#' where \eqn{B_l} are random matrices with orthonormal columns (obtained from
#' the singular value decomposition of matrices with \eqn{U(-1, 1)} entries)
#' and \eqn{\omega} is an elementwise nonlinearity. With one layer the mixing
#' is linear. After each layer the data are divided by the mean standard
#' deviation of the variables if `normalize = TRUE`. The first matrix maps
#' the data to `output_dim` dimensions; `output_dim` larger than the number
#' of columns of `x` gives overdetermined mixtures.
#'
#' The available nonlinearities are `"elu"` (\eqn{x} for \eqn{x \ge 0},
#' \eqn{e^x - 1} otherwise), `"lrelu"` (leaky ReLU with slope 0.2),
#' `"xtanh"` (\eqn{\tanh(x) + 0.1x}), `"smooth_lrelu"`, `"exp_arctan"`,
#' `"leaky_softplus"` and `"logsumexp"`.
#'
#' `final_activation = "exp"` or `"softplus"` makes the output positive, for
#' example to be used as Poisson rates.
#'
#' @param x A numeric matrix of latent components, one column per component.
#' @param n_layers Number of mixing layers.
#' @param nonlinearity The nonlinearity between layers.
#' @param output_dim Dimension of the mixed data, at least `ncol(x)`.
#' @param final_activation Activation applied to the output: `"none"`,
#'   `"exp"` or `"softplus"`.
#' @param normalize Logical. Scale the data after each layer?
#' @return A numeric matrix with `output_dim` columns.
#' @seealso [simulate_sources()]
#' @examples
#' z <- matrix(rnorm(300), ncol = 3)
#' x <- mix_data(z, n_layers = 3)
#' x5 <- mix_data(z, n_layers = 2, output_dim = 5)
#'
#' # Poisson counts
#' rates <- mix_data(z, n_layers = 2, final_activation = "softplus")
#' counts <- matrix(rpois(length(rates), rates), nrow = nrow(rates))
#' @export
mix_data <- function(x, n_layers = 1,
                     nonlinearity = c("elu", "lrelu", "xtanh", "smooth_lrelu",
                                      "exp_arctan", "leaky_softplus",
                                      "logsumexp"),
                     output_dim = ncol(x),
                     final_activation = c("none", "exp", "softplus"),
                     normalize = TRUE,
                     normalize_sd = 1) {
  nonlinearity <- match.arg(nonlinearity)
  final_activation <- match.arg(final_activation)
  x <- as.matrix(x)
  if (output_dim < ncol(x)) {
    stop("`output_dim` must be at least ncol(x) for an injective mixing.",
         call. = FALSE)
  }
  omega <- switch(nonlinearity,
    elu = function(a) ifelse(a < 0, exp(a) - 1, a),
    lrelu = function(a) ifelse(a < 0, 0.2 * a, a),
    xtanh = function(a) tanh(a) + 0.2 * a,
    smooth_lrelu = function(a) 0.5 * (1.05 * a + 0.95 * sqrt(a^2 + 0.1)),
    exp_arctan = function(a) 0.5 * a * exp(atan(a)),
    leaky_softplus = function(a) 0.1 * a + 0.9 * log1p(exp(a)),
    logsumexp = function(a) log(exp(0.1 * a) + exp(a)) - log(2)
  )
  for (l in seq_len(n_layers)) {
    B <- svd(matrix(runif(output_dim * ncol(x), -1, 1), output_dim))$u
    x <- x %*% t(B)
    if (normalize) x <- normalize_sd * x / mean(apply(x, 2, sd))
    if (l < n_layers) x <- omega(x)
  }
  x <- switch(final_activation,
    none = x,
    exp = exp(x),
    softplus = log1p(exp(x))
  )
  colnames(x) <- paste0("X", seq_len(ncol(x)))
  x
}

generate_fine_layer <- function(n_fine, domain = c(0, 1, 0, 1)) {
  if (length(domain) == 2L) {
    x_range <- y_range <- domain
  } else if (length(domain) == 4L) {
    x_range <- domain[1:2]
    y_range <- domain[3:4]
  } else {
    stop("`domain` must have length 2 or 4.", call. = FALSE)
  }

  pts <- cbind(runif(n_fine, x_range[1], x_range[2]),
               runif(n_fine, y_range[1], y_range[2]))
  del <- deldir::deldir(pts[, 1], pts[, 2], rw = c(x_range[1], x_range[2],
                                                   y_range[1], y_range[2]))
  edges <- as.matrix(del$delsgs[, c("ind1", "ind2")])
  g <- igraph::simplify(igraph::graph_from_edgelist(edges, directed = FALSE))
  list(pts = pts, del = del, graph = g)
}

place_seeds_fixed <- function(fine_pts, n_s,
                              seed_type = c("uniform", "clustered"),
                              n_clusters = 10, cluster_sd = 0.04,
                              xy_coef = 1) {
  seed_type <- match.arg(seed_type)
  n_fine <- nrow(fine_pts)
  k <- min(20L, n_fine)

  if (seed_type == "uniform") {
    seed_coords <- matrix(runif(2 * n_s), ncol = 2)
  } else {
    centres <- matrix(runif(2 * n_clusters), ncol = 2)
    cluster_id <- sample.int(n_clusters, n_s, replace = TRUE)
    seed_coords <- centres[cluster_id, , drop = FALSE] +
      cbind(rnorm(n_s, sd = cluster_sd),
            rnorm(n_s, sd = xy_coef * cluster_sd))
    seed_coords <- pmax(pmin(seed_coords, 1), 0)
  }

  nn <- t(vapply(seq_len(n_s), function(i) {
    d2 <- rowSums((sweep(fine_pts, 2L, seed_coords[i, ], "-"))^2)
    order(d2)[seq_len(min(k, n_fine))]
  }, integer(min(k, n_fine))))

  assigned <- integer(n_s)
  used <- logical(n_fine)
  for (i in seq_len(n_s)) {
    cands <- nn[i, ][!used[nn[i, ]]]
    k_i <- k
    while (length(cands) == 0L) {
      k_i <- k_i * 2L
      if (k_i > n_fine) {
        stop("Fine layer too small relative to `n_s`; increase `n_fine`.",
             call. = FALSE)
      }
      d2 <- rowSums((sweep(fine_pts, 2L, seed_coords[i, ], "-"))^2)
      nn_i <- order(d2)[seq_len(k_i)]
      cands <- nn_i[!used[nn_i]]
    }
    assigned[i] <- cands[1L]
    used[cands[1L]] <- TRUE
  }
  assigned
}

assign_graph_voronoi <- function(g, seed_idx) {
  dist_mat <- igraph::distances(g, v = seed_idx, to = igraph::V(g), weights = NA)
  apply(dist_mat, 2L, which.min)
}

build_areal_adjacency <- function(g, cluster_id) {
  edge_df <- igraph::as_data_frame(g, what = "edges")
  if (is.null(edge_df) || nrow(edge_df) == 0L) {
    return(matrix(0, length(unique(cluster_id)), length(unique(cluster_id))))
  }
  c1 <- as.integer(cluster_id[as.integer(edge_df$from)])
  c2 <- as.integer(cluster_id[as.integer(edge_df$to)])
  cross <- c1 != c2
  pairs <- unique(data.frame(a = pmin(c1[cross], c2[cross]),
                            b = pmax(c1[cross], c2[cross])))
  n_areas <- length(unique(cluster_id))
  W <- matrix(0, n_areas, n_areas)
  if (nrow(pairs) > 0L) {
    W[cbind(pairs$a, pairs$b)] <- 1
    W[cbind(pairs$b, pairs$a)] <- 1
  }
  W
}

build_areal_polygons <- function(fine, cluster_id) {
  if (!requireNamespace("sf", quietly = TRUE)) {
    stop("The `sf` package is required to build areal polygons.", call. = FALSE)
  }
  if (!requireNamespace("deldir", quietly = TRUE)) {
    stop("The `deldir` package is required to build areal polygons.", call. = FALSE)
  }

  tiles <- deldir::tile.list(fine$del)
  poly_list <- lapply(seq_along(tiles), function(i) {
    tl <- tiles[[i]]
    coords <- cbind(tl$x, tl$y)
    coords <- rbind(coords, coords[1L, , drop = FALSE])
    sf::st_polygon(list(coords))
  })

  voronoi_sf <- sf::st_sf(cluster = cluster_id,
                          geometry = sf::st_sfc(poly_list))
  areas_list <- lapply(split(voronoi_sf, voronoi_sf$cluster), function(g) {
    sf::st_sf(cluster = unique(g$cluster),
              geometry = sf::st_sfc(sf::st_union(sf::st_geometry(g))))
  })
  areas_sf <- do.call(rbind, areas_list)
  sf::st_make_valid(areas_sf)
}

#' Generate a Spatial Areal Partition from a Fine Grid
#'
#' Constructs a set of areal regions by (i) generating a dense fine-grained point
#' layer over the unit square, (ii) placing `n_area` seed points, (iii)
#' assigning each fine point to the nearest seed using the graph induced by the
#' fine layer, and (iv) deriving the adjacency matrix and geometry of the final
#' regions. This yields an areal graph that matches the graph-based Voronoi
#' construction used for areal source simulation and auxiliary-data calculations.
#'
#' When both `deldir` and `igraph` are available, the fine layer is linked by a
#' Delaunay graph and the region assignment is obtained from graph distances. If
#' those packages are absent, a fallback radius graph is used instead.
#'
#' @param n_area Integer, the desired number of areal units.
#' @param n_fine Integer, the number of fine grid points used to define the
#'   partition. If `NULL`, a default value is chosen as `max(200, 30 * n_area)`.
#' @param seed_type Seed placement scheme: `"uniform"` or `"clustered"`.
#' @param n_clusters Number of clusters for `seed_type = "clustered"`.
#' @param cluster_sd Standard deviation used for the within-cluster seed jitter.
#' @param xy_coef Additional anisotropy parameter for clustered seed placement.
#' @param radius Optional radius used in the fallback adjacency construction.
#'   If `NULL`, it is chosen from the centroid spacing of the generated regions.
#' @param seed Optional random seed for reproducible output.
#' @return A list with the elements:
#'   * `fine_points`: coordinates of the fine grid.
#'   * `seed_points`: coordinates of the seed points.
#'   * `area_id`: region assignment for each fine point.
#'   * `area_coords`: centroid coordinates of the areal units.
#'   * `W`: binary adjacency matrix for the areal regions.
#'   * `area_centres`: same as `area_coords`.
#'   * `areas_sf`: polygon geometry for each areal region when the `sf` and
#'     `deldir` packages are available.
#'   * `cluster_id`: final region id attached to the fine points.
#'   * `fine`: the internal fine-layer object used to build the partition.
#' @examples
#' areal <- generate_areal_units(6, n_fine = 200, seed_type = "uniform",
#'                               seed = 1)
#' dim(areal$W)
#' @seealso [simulate_sources_areal()]
#' @export
generate_areal_units <- function(n_area, n_fine = NULL,
                                seed_type = c("uniform", "clustered"),
                                n_clusters = 3L, cluster_sd = 0.04,
                                xy_coef = 1, radius = NULL,
                                seed = NULL) {
  seed_type <- match.arg(seed_type)
  if (!is.null(seed)) set.seed(seed)
  if (length(n_area) != 1L || n_area < 2L) {
    stop("`n_area` must be a single integer greater than or equal to 2.",
         call. = FALSE)
  }
  if (is.null(n_fine)) n_fine <- max(200L, 30L * n_area)
  n_fine <- as.integer(n_fine)

  if (requireNamespace("deldir", quietly = TRUE) &&
      requireNamespace("igraph", quietly = TRUE)) {
    fine <- generate_fine_layer(n_fine)
  } else {
    fine <- list(pts = matrix(runif(2 * n_fine), ncol = 2))
  }

  seed_idx <- place_seeds_fixed(fine$pts, n_area, seed_type = seed_type,
                               n_clusters = n_clusters, cluster_sd = cluster_sd,
                               xy_coef = xy_coef)

  cluster_id <- assign_graph_voronoi(fine$graph, seed_idx)
  cluster_id <- as.integer(factor(cluster_id))
  n_area_actual <- length(unique(cluster_id))
  if (n_area_actual < 2L) {
    stop("The generated areal graph does not contain enough valid regions.",
         call. = FALSE)
  }

  W <- build_areal_adjacency(fine$graph, cluster_id)
  area_centres <- t(vapply(seq_len(n_area_actual), function(i) {
    colMeans(fine$pts[cluster_id == i, , drop = FALSE])
  }, numeric(2L)))

  if (nrow(W) == 0L) {
    W <- matrix(0, n_area_actual, n_area_actual)
  }
  if (is.null(radius)) {
    if (n_area_actual > 1L) {
      dnn <- min(as.matrix(dist(area_centres)))
      radius <- if (is.finite(dnn) && dnn > 0) 1.5 * dnn else 0.2
    } else {
      radius <- 0.2
    }
  }

  if (sum(W) == 0L) {
    D <- as.matrix(dist(area_centres))
    W <- ((D > 0) & (D <= radius)) * 1
    diag(W) <- 0
  }

  if (is.null(colnames(W))) colnames(W) <- rownames(W) <- paste0("A", seq_len(n_area_actual))
  areas_sf <- build_areal_polygons(fine, cluster_id)

  list(fine_points = fine$pts,
       seed_points = fine$pts[seed_idx, , drop = FALSE],
       area_id = cluster_id,
       area_coords = area_centres,
       W = W,
       area_centres = area_centres,
       areas_sf = areas_sf,
       cluster_id = cluster_id,
       fine = fine)
}

#' Simulate Nonstationary Areal Sources Over Time
#'
#' Simulates latent sources for areal units over time from a nonstationary
#' spatial autoregressive (SAR) model with autoregressive time dependence. This
#' is the intended areal analogue of [simulate_sources()] for spatially
#' structured data. Replaces the earlier version of this function.
#'
#' @details
#' For each component and time point the innovation is
#' \deqn{u_t = (I - \mathrm{diag}(\beta) W)^{-1} \epsilon_t,\quad
#' \epsilon_{it} \sim N(0, \sigma_i^2 m_t),}{
#' u_t = (I - diag(beta) W)^(-1) eps_t, eps_it ~ N(0, sigma_i^2 m_t),}
#' where `W` is the row-standardised adjacency matrix, and the component is
#' \eqn{z_t = \gamma_t z_{t-1} + u_t}{z_t = gamma_t z_(t-1) + u_t}. Three
#' kinds of nonstationarity can be present:
#'
#' * **Spatial variance** \eqn{\sigma_i}: built from one leading eigenvector
#'   of the areal graph (see `basis`), scaled to `sd_range`.
#' * **Temporal variance** \eqn{m_t}: block-wise, drawn separately for each
#'   component, with `n_segments_time` blocks and values in `sd_range`.
#' * **Spatial dependence** \eqn{\beta_i}: constant over areas by default, with
#'   one value per component drawn from `beta_s_range`. With
#'   `nonstat_dependence = TRUE` it varies between areas: another leading
#'   eigenvector of the graph (different from the one used for the variance
#'   when several are available) is scaled to a random sub-interval of
#'   `beta_s_range`, so that the strength of the spatial dependence changes
#'   smoothly over the map. Because `W` is row-standardised and all
#'   \eqn{|\beta_i| < 1}, the system is always invertible.
#'
#' The AR coefficients \eqn{\gamma_t} are drawn independently for each time
#' point from `ar_range`.
#'
#' @param p Number of latent components.
#' @param areal Optional output from [generate_areal_units()].
#' @param W Optional adjacency matrix.
#' @param n_area Number of areal units when `W` and `areal` are not supplied.
#' @param n_fine Number of fine points used to generate areal units.
#' @param n_time Number of time points.
#' @param n_segments_time Number of temporal variance blocks.
#' @param sd_range Range of the spatial and temporal standard deviations used
#'   to scale the eigenvector-based spatial variance and the time-block
#'   variance.
#' @param ar_range Range for the AR(1) coefficients.
#' @param beta_s_range Range (inside (-1, 1)) of the spatial dependence
#'   parameter. For stationary dependence one value per component is drawn from
#'   it; for `nonstat_dependence = TRUE` it bounds the range over which
#'   \eqn{\beta_i} varies.
#' @param beta_base_mult_range Range of the multiplicative factor for the base
#'   value of the spatial dependence parameter when `nonstat_dependence = TRUE`.
#' @param K Positive integer giving the number of basis functions to 
#' create nonstationarity.
#' @param seed_type Type of seed placement when generating areal units.
#' @param n_clusters Number of clusters when `seed_type = "clustered"`.
#' @param cluster_sd Cluster spread used in seed generation.
#' @param xy_coef Additional anisotropy in the clustered seed layout.
#' @param radius Radius used in the fallback graph construction.
#' @param seed Optional random seed.
#' @param nonstat_dependence Logical. Let the spatial dependence parameter vary
#'   between areas, driven by a graph eigenvector?
#' @param nonstat_dependence_time Logical. Together with
#'   `nonstat_dependence = TRUE`, draw a separate eigenvector index for each
#'   temporal segment (`n_segments_time`), so that the spatial dependence
#'   pattern also changes in time.
#' @param basis Eigenvectors used for the spatial patterns of the variance and
#'   the dependence: `"moran"` (positive Moran eigenvector maps) or
#'   `"laplacian"` (smoothest non-trivial graph Laplacian eigenvectors).
#' @return A list with elements `sources`, `labels`, `W`, `coords`, `time`,
#'   `area`, `group`, `areal`, `mem` (the eigenvectors of the chosen basis),
#'   `basis`, and `beta_s`, an `n_area x p` matrix with the dependence
#'   parameter of every area and component (constant over the areas unless
#'   `nonstat_dependence = TRUE`; averaged over time if
#'   `nonstat_dependence_time = TRUE`), and `beta_s_time`, an
#'   `n_area x n_time x p` array with the dependence parameter at each time.
#' @examples
#' areal <- generate_areal_units(8, n_fine = 200, seed = 1)
#' src <- simulate_sources_areal(3, areal = areal, n_time = 20,
#'                                n_segments_time = 3)
#' dim(src$sources)
#'
#' # dependence strength varying over space
#' src2 <- simulate_sources_areal(3, areal = areal, n_time = 20,
#'                                nonstat_dependence = TRUE, basis = "laplacian")
#' round(src2$beta_s, 2)
#'
#' # dependence pattern also changing between temporal segments
#' src3 <- simulate_sources_areal(3, areal = areal, n_time = 20,
#'                                nonstat_dependence = TRUE,
#'                                nonstat_dependence_time = TRUE)
#' @export
simulate_sources_areal <- function(p, areal = NULL, W = NULL, n_area = NULL,
                                   n_fine = NULL, n_time = 50,
                                   n_segments_time = 5,
                                   sd_range = c(0.1, 5),
                                   ar_range = c(0.3, 0.9),
                                   beta_s_range = c(0.3, 0.9),
                                   beta_base_mult_range = c(0.75, 1),
                                   K = 10L,
                                   seed_type = c("uniform", "clustered"),
                                   n_clusters = 3L, cluster_sd = 0.04,
                                   xy_coef = 1, radius = NULL,
                                   seed = NULL,
                                   nonstat_dependence = FALSE,
                                   nonstat_dependence_time = FALSE,
                                   basis = c("moran", "laplacian")) {
  seed_type <- match.arg(seed_type)
  basis <- match.arg(basis)
  if (!is.null(seed)) set.seed(seed)

  if (is.null(areal)) {
    if (!is.null(W)) {
      areal <- list(W = as.matrix(W), area_coords = NULL, area_id = seq_len(nrow(W)))
    } else {
      if (is.null(n_area)) n_area <- 8L
      areal <- generate_areal_units(n_area = n_area, n_fine = n_fine,
                                    seed_type = seed_type,
                                    n_clusters = n_clusters,
                                    cluster_sd = cluster_sd,
                                    xy_coef = xy_coef,
                                    radius = radius, seed = seed)
    }
  }

  W <- as.matrix(areal$W)
  if (nrow(W) != ncol(W)) stop("`W` must be square.", call. = FALSE)
  n_area <- nrow(W)
  if (n_time < 1L) stop("`n_time` must be positive.", call. = FALSE)
  if (n_segments_time < 1L) stop("`n_segments_time` must be positive.", call. = FALSE)
  n_segments_time <- min(as.integer(n_segments_time), as.integer(n_time))

  if (length(sd_range) != 2L || any(!is.finite(sd_range)) || any(sd_range <= 0)) {
    stop("`sd_range` must be a numeric vector of length 2 with positive finite values.",
         call. = FALSE)
  }
  sd_range <- sort(sd_range)
  if (length(ar_range) != 2L || any(!is.finite(ar_range))) ar_range <- c(0.3, 0.9)
  if (length(beta_s_range) != 2L || any(!is.finite(beta_s_range)) ||
      any(abs(beta_s_range) >= 1)) {
    stop("`beta_s_range` must be a numeric vector of length 2 inside (-1, 1).",
         call. = FALSE)
  }
  beta_s_range <- sort(beta_s_range)

  # Row-standardised weights (zero rows stay zero)
  rs <- rowSums(W)
  W_row <- W / ifelse(rs > 0, rs, 1)

  # Eigenvector basis for the spatial variance and dependence patterns
  basis_mat <- if (basis == "moran") {
    compute_MEM_vectors(W, K = min(K, max(1L, n_area - 1L)),
                        positive_only = TRUE)$mem
  } else {
    compute_laplacian_eigenvectors(as_adj_list(W),
                                   K = min(K, max(1L, n_area - 1L)))$phi
  }
  K_basis <- ncol(basis_mat)
  if (K_basis < 1L) {
    stop("No eigenvectors were found for the supplied areal graph.", call. = FALSE)
  }
  scale01 <- function(v) {
    v <- v - min(v)
    if (max(v) > 0) v / max(v) else v
  }

  # Temporal block assignment shared by the variance and AR profiles.
  draw_time_labels <- function() {
    if (n_time == 1L) return(rep(1L, n_time))
    centres <- sample(seq_len(n_time), n_segments_time)
    vapply(seq_len(n_time),
           function(t) which.min(abs(centres - t)), integer(1L))
  }

  z <- matrix(NA_real_, n_area * n_time, p)
  beta_s_used <- matrix(NA_real_, n_area, p)
  beta_s_time <- array(NA_real_, c(n_area, n_time, p))
  I_n <- diag(n_area)

  for (j in seq_len(p)) {
    base_beta_coef <- runif(1L, beta_base_mult_range[1], beta_base_mult_range[2])
    # ---- spatial variance pattern from one leading eigenvector -----------
    idx_var <- sample.int(K_basis, 1L)
    sigma_i <- scale01(as.numeric(basis_mat[, idx_var])) * diff(sd_range) + sd_range[1]

    # ---- temporal variance and AR profiles share the same segment labels -
    time_labels <- draw_time_labels()
    time_mult <- runif(n_segments_time, sd_range[1], sd_range[2])[time_labels]
    
    # ---- spatial dependence: constant, or driven by eigenvectors ---------
    # With `nonstat_dependence_time`, every temporal segment draws its own
    # eigenvector index, so the dependence pattern changes between segments.
    b <- beta_s_range
    others <- setdiff(seq_len(K_basis), idx_var)
    if (length(others) == 0L) others <- idx_var
    pick <- function(n) others[sample.int(length(others), n, replace = n > length(others))]
    if (nonstat_dependence && nonstat_dependence_time) {
      seg_idx <- pick(n_segments_time)
      seg_beta <- vapply(seg_idx, function(k) {
        b[1] + (b[2] - b[1]) * scale01(as.numeric(basis_mat[, k]))
      }, numeric(n_area))
      seg_beta <- matrix(seg_beta, n_area, n_segments_time)
      # Scale betas from both ends according to base_beta_coef
      seg_beta <- seg_beta * base_beta_coef + (1 - base_beta_coef) / 2
    } else if (nonstat_dependence) {
      idx_dep <- pick(1L)
      seg_beta <- matrix(b[1] + (b[2] - b[1]) * scale01(as.numeric(basis_mat[, idx_dep])),
                         n_area, n_segments_time)
    } else {
      seg_beta <- matrix(runif(1L, b[1], b[2]), n_area, n_segments_time)
    }
    beta_s_time[, , j] <- seg_beta[, time_labels, drop = FALSE]
    beta_s_used[, j] <- rowMeans(beta_s_time[, , j, drop = FALSE])
    # (I - diag(beta) W_row)^{-1}; beta_i * W_row scales row i by beta_i
    A_inv_seg <- lapply(seq_len(n_segments_time), function(g) {
      solve(I_n - seg_beta[, g] * W_row)
    })

    ar_coeff <- runif(n_segments_time, min(ar_range), max(ar_range))[time_labels]

    # ---- innovations: heteroscedastic noise, then SAR filtering ----------
    sd_mat  <- outer(sigma_i, sqrt(time_mult))                 # n_area x n_time
    eps_mat <- matrix(rnorm(n_area * n_time, sd = sd_mat), n_area, n_time)
    u_mat   <- eps_mat                                         # SAR on innovation only
    for (g in seq_len(n_segments_time)) {
      cols <- which(time_labels == g)
      if (length(cols)) u_mat[, cols] <- A_inv_seg[[g]] %*% eps_mat[, cols, drop = FALSE]
    }

    # ---- plain AR(1) recursion over time ---------------------------------
    z_t <- matrix(0, n_area, n_time)
    z_t[, 1L] <- u_mat[, 1L]
    if (n_time > 1L) {
      for (t in 2:n_time) z_t[, t] <- ar_coeff[t] * z_t[, t - 1L] + u_mat[, t]
    }
    z[, j] <- as.numeric(z_t)
  }

  area_order <- rep(seq_len(n_area), n_time)
  time_order <- rep(seq_len(n_time), each = n_area)
  colnames(z) <- paste0("Z", seq_len(p))
  colnames(beta_s_used) <- colnames(z)
  coords <- if (is.null(areal$area_coords)) matrix(runif(2 * n_area), ncol = 2)
            else areal$area_coords
  coords <- coords[area_order, , drop = FALSE]

  list(sources = z,
       labels = time_order,
       W = W,
       coords = coords,
       time = time_order,
       area = area_order,
       group = area_order,
       areal = areal,
       mem = basis_mat,
       basis = basis,
       beta_s = beta_s_used,
       beta_s_time = beta_s_time)
}

#' Absolute Mean Correlation Coefficient
#'
#' Computes the mean absolute correlation coefficient (MCC) between estimated
#' and true components, a common performance measure in nonlinear BSS.
#'
#' The MCC is
#' \deqn{MCC(K) = \frac{1}{p} \max_{P} \mathrm{tr}(P |K|),}{
#' MCC(K) = (1/p) max_P tr(P |K|),}
#' where \eqn{K} is the \eqn{p \times p} correlation matrix, \eqn{|K|} its
#' elementwise absolute value and the maximum is over permutation matrices
#' \eqn{P}. The maximisation is a linear assignment problem, solved with the
#' Hungarian algorithm. For a non-square matrix, \eqn{p} is the smaller
#' dimension.
#'
#' @param x A correlation matrix, or a matrix of estimated components.
#' @param y Optional matrix of true components. If given, the correlation
#'   matrix `cor(x, y)` is used.
#' @return The MCC, a number between 0 and 1.
#' @examples
#' z <- matrix(rnorm(300), ncol = 3)
#' z_hat <- z[, c(2, 3, 1)] + matrix(rnorm(300, sd = 0.3), ncol = 3)
#' absolute_mean_correlation(z_hat, z)
#' absolute_mean_correlation(cor(z_hat, z))
#' @export
absolute_mean_correlation <- function(x, y = NULL) {
  K <- if (is.null(y)) as.matrix(x) else cor(as.matrix(x), as.matrix(y))
  if (nrow(K) > ncol(K)) K <- t(K)
  -RcppHungarian::HungarianSolver(-abs(K))$cost / min(dim(K))
}

#' Simulate Areal Sources from a Directed Acyclic Spatial Prior
#'
#' Simulates latent components over areal units and time from a spatial
#' autoregression on a directed acyclic version of the areal graph: visiting
#' the areas in a fixed order, each area depends on the average of its
#' already visited neighbours,
#' \deqn{z_{i} = \beta_i \cdot \mathrm{mean}_{j \in pa(i)} z_{j} + \sigma_i \epsilon_i,}{
#' z_i = beta_i * mean_{j in pa(i)} z_j + sigma_i * eps_i,}
#' with independent standard normal \eqn{\epsilon_i} for every area and time
#' point. Because the graph is acyclic, this is an exact joint model whose
#' conditional distributions are the Gaussian conditionals fitted by [GiVAE()]
#' with the graph returned in `graph`, so the model matches the data
#' generating process exactly. Time points are independent replicates (there is
#' no temporal dependence).
#'
#' The coupling \eqn{\beta_i} of each component varies over the areas: it is
#' one eigenvector of the areal graph (see `basis`) scaled to `beta_s_range`.
#' The variance \eqn{\sigma_i^2} is constant unless `sd_range` has two
#' different values; then it follows another eigenvector.
#'
#' @param p Number of components.
#' @param areal Output of [generate_areal_units()].
#' @param n_time Number of time points (independent replicates).
#' @param ordering Optional numeric vector with one value per area giving the
#'   order in which the areas are visited (smaller first). Defaults to the
#'   x-coordinate of the area centres.
#' @param sd_range Range of the innovation standard deviations.
#' @param beta_s_range Range (inside (-1, 1)) of the coupling.
#' @param basis Eigenvectors used for the spatial patterns: `"laplacian"` or
#'   `"moran"`. With `"laplacian"` the coupling of component `j` is the scaled
#'   `j`-th column of [aux_laplacian()] of the same graph.
#' @param seed Optional random seed.
#' @return A list like [simulate_sources_areal()] with `sources`, `time`,
#'   `area`, `group`, `coords`, `W`, `areal`, `mem`, `basis` and `beta_s`
#'   (an `n_area x p` matrix of the couplings), plus `graph` (the directed
#'   acyclic [nlbss_graph()] of the parents) and `ordering`.
#' @examples
#' areal <- generate_areal_units(20, n_fine = 400, seed = 1)
#' src <- simulate_sources_dag(3, areal, n_time = 50, seed = 1)
#' src$graph
#' @seealso [nlbss_dag()], [simulate_sources_areal()], [GiVAE()]
#' @export
simulate_sources_dag <- function(p, areal, n_time = 100, ordering = NULL,
                                 sd_range = c(1, 1), beta_s_range = c(0.1, 0.95),
                                 basis = c("laplacian", "moran"), seed = NULL) {
  basis <- match.arg(basis)
  if (!is.null(seed)) set.seed(seed)
  W <- as.matrix(areal$W)
  n_area <- nrow(W)
  if (length(beta_s_range) != 2L || any(!is.finite(beta_s_range)) ||
      any(abs(beta_s_range) >= 1)) {
    stop("`beta_s_range` must be a numeric vector of length 2 inside (-1, 1).",
         call. = FALSE)
  }
  if (length(sd_range) != 2L || any(!is.finite(sd_range)) || any(sd_range <= 0)) {
    stop("`sd_range` must be a numeric vector of length 2 with positive values.",
         call. = FALSE)
  }
  beta_s_range <- sort(beta_s_range)
  sd_range <- sort(sd_range)
  if (is.null(ordering)) {
    if (is.null(areal$area_coords)) {
      stop("Give `ordering`: `areal` has no area coordinates.", call. = FALSE)
    }
    ordering <- areal$area_coords[, 1]
  }

  graph <- nlbss_dag(W, order = ordering, name = "space (DAG)")
  parents <- graph$adj_list
  visit <- sort.list(rank(ordering, ties.method = "first"))   # areas in visiting order

  K <- min(10L, max(1L, n_area - 1L))
  basis_mat <- if (basis == "moran") {
    compute_MEM_vectors(W, K = K, positive_only = TRUE)$mem
  } else {
    compute_laplacian_eigenvectors(as_adj_list(W), K = K)$phi
  }
  K_basis <- ncol(basis_mat)
  scale01 <- function(v) {
    v <- v - min(v)
    if (max(v) > 0) v / max(v) else v
  }

  z <- matrix(NA_real_, n_area * n_time, p)
  beta_used <- matrix(NA_real_, n_area, p)
  for (j in seq_len(p)) {
    idx <- ((j - 1L) %% K_basis) + 1L
    beta_i <- beta_s_range[1] + diff(beta_s_range) * scale01(basis_mat[, idx])
    sigma_i <- if (diff(sd_range) == 0) {
      rep(sd_range[1], n_area)
    } else {
      idx_var <- (idx %% K_basis) + 1L
      sd_range[1] + diff(sd_range) * scale01(basis_mat[, idx_var])
    }
    beta_used[, j] <- beta_i

    Z <- matrix(0, n_area, n_time)
    for (i in visit) {
      pa <- parents[[i]]
      m <- if (length(pa)) colMeans(Z[pa, , drop = FALSE]) else 0
      Z[i, ] <- beta_i[i] * m + rnorm(n_time, sd = sigma_i[i])
    }
    z[, j] <- as.numeric(Z)          # area varies fastest within time
  }

  area_order <- rep(seq_len(n_area), n_time)
  time_order <- rep(seq_len(n_time), each = n_area)
  colnames(z) <- paste0("Z", seq_len(p))
  colnames(beta_used) <- colnames(z)
  coords <- if (is.null(areal$area_coords)) matrix(NA_real_, n_area, 2) else areal$area_coords
  list(sources = z,
       labels = time_order,
       W = W,
       coords = coords[area_order, , drop = FALSE],
       time = time_order,
       area = area_order,
       group = area_order,
       areal = areal,
       mem = basis_mat,
       basis = basis,
       beta_s = beta_used,
       graph = graph,
       ordering = ordering)
}