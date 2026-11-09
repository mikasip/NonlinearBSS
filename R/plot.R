# ggplot2 helpers shared by the plot methods of the package.

theme_nlbss <- function(base_size = 11) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      panel.grid.minor = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold", hjust = 0),
      plot.title = ggplot2::element_text(face = "bold"),
      legend.title = ggplot2::element_text(size = ggplot2::rel(0.9))
    )
}

# Spatial layer of values on points or polygons.
# `values`: n x k matrix (or vector) with one column per panel; `geometry`: an
# `sfc` with n elements or NULL, in which case `xy` (n x 2) gives point
# locations.
map_plot <- function(values, geometry = NULL, xy = NULL, labels = NULL,
                     fill = "value", ncol = NULL, palette = "viridis",
                     border = "white", ...) {
  values <- as.matrix(values)
  n <- nrow(values)
  k <- ncol(values)
  if (is.null(labels)) {
    labels <- if (is.null(colnames(values))) paste("Component", seq_len(k)) else colnames(values)
  }
  panel <- factor(rep(labels, each = n), levels = unique(labels))
  value <- as.vector(values)
  scale_fn <- function(aesthetic) {
    ggplot2::scale_colour_viridis_c(name = fill, option = palette, aesthetics = aesthetic)
  }

  if (!is.null(geometry)) {
    require_pkg("sf")
    geom_rep <- geometry[rep(seq_len(n), k)]
    d <- sf::st_sf(data.frame(value = value, panel = panel), geometry = geom_rep)
    polygons <- all(sf::st_geometry_type(geometry) %in%
                      c("POLYGON", "MULTIPOLYGON"))
    p <- ggplot2::ggplot(d)
    if (polygons) {
      p <- p + ggplot2::geom_sf(ggplot2::aes(fill = .data$value),
                                colour = border, linewidth = 0.1, ...) +
        scale_fn("fill")
    } else {
      p <- p + ggplot2::geom_sf(ggplot2::aes(colour = .data$value), ...) +
        scale_fn("colour")
    }
  } else {
    d <- data.frame(x = rep(xy[, 1], k), y = rep(xy[, 2], k),
                    value = value, panel = panel)
    p <- ggplot2::ggplot(d, ggplot2::aes(.data$x, .data$y,
                                          colour = .data$value)) +
      ggplot2::geom_point(...) + scale_fn("colour") +
      ggplot2::coord_fixed() + ggplot2::labs(x = NULL, y = NULL)
  }
  p + ggplot2::facet_wrap(ggplot2::vars(.data$panel), ncol = ncol) +
    theme_nlbss() +
    ggplot2::theme(axis.text = ggplot2::element_blank(),
                   panel.grid = ggplot2::element_blank())
}

#' Plot Values on Areal Units
#'
#' Draws one or several variables (for example simulated sources or estimated
#' components) as a choropleth map of the areal units, one panel per column.
#'
#' @param areal Output of [generate_areal_units()] (needs the `sf` package), or
#'   an `sf` object with polygons.
#' @param values A vector with one value per areal unit, or a matrix with one
#'   column per panel.
#' @param labels Panel titles. Defaults to the column names.
#' @param fill Title of the colour legend.
#' @param ncol Number of panel columns.
#' @param palette Viridis palette option, see [ggplot2::scale_fill_viridis_c()].
#' @param ... Further arguments passed to [ggplot2::geom_sf()].
#' @return A `ggplot` object.
#' @examples
#' \donttest{
#' if (requireNamespace("sf", quietly = TRUE)) {
#'   areal <- generate_areal_units(30, n_fine = 300, seed = 1)
#'   src <- simulate_sources_areal(2, areal = areal, n_time = 5, seed = 1)
#'   plot_areal(areal, src$sources[src$time == 1, ])
#' }
#' }
#' @export
plot_areal <- function(areal, values, labels = NULL, fill = "value",
                       ncol = NULL, palette = "viridis", ...) {
  require_pkg("sf")
  geometry <- if (inherits(areal, c("sf", "sfc"))) sf::st_geometry(areal) else areal$areas_sf
  if (is.null(geometry)) {
    stop("`areal` has no polygons; the `sf` and `deldir` packages are needed.",
         call. = FALSE)
  }
  geometry <- sf::st_geometry(geometry)
  values <- as.matrix(values)
  if (nrow(values) != length(geometry)) {
    stop("`values` must have one row per areal unit (", length(geometry), ").",
         call. = FALSE)
  }
  map_plot(values, geometry = geometry, labels = labels, fill = fill,
           ncol = ncol, palette = palette, ...)
}

# Geometry of the data rows for map plots: polygons or points of sf data,
# or the sp geometry of STFDF data, NULL for plain coordinates.
row_geometry <- function(data, rows) {
  g <- data$template$geometry
  if (!is.null(g)) return(g[rows])
  sp_geom <- data$template$sp
  if (!is.null(sp_geom) && requireNamespace("sf", quietly = TRUE)) {
    g <- sf::st_as_sfc(sp_geom)
    return(g[(rows - 1L) %% length(g) + 1L])
  }
  NULL
}
