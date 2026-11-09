method_title <- function(method) {
  switch(method,
    TCL = "Time contrastive learning (TCL)",
    VAE = "Variational autoencoder (VAE)",
    iVAE = "Identifiable variational autoencoder (iVAE)",
    iVAEar = "Identifiable variational autoencoder with autoregressive components (iVAEar)",
    GiVAE = "Graph identifiable variational autoencoder (GiVAE)"
  )
}

#' Extract the Estimated Latent Components
#'
#' Returns the estimated latent components of a fitted model in the format of
#' the data the model was fitted to: for example a `ts` object for a `ts`
#' input, an `sf` object with the original geometries for an `sf` input, and
#' a matrix for a matrix input.
#'
#' As in other blind source separation packages, the standardised components
#' are also stored as the element `S` of the fitted object, so that for
#' example `JADE::bss.components()` works as well. `components()` is a method
#' for the generic of the \pkg{generics} package, and it is also registered
#' for the `components()` generic of \pkg{ICS} when that package is loaded.
#'
#' @param object A fitted model from [iVAE()], [iVAEar()], [VAE()] or [TCL()].
#' @param format `"input"` for the input format, `"matrix"` for a matrix.
#' @param unscaled Logical. Return the unstandardised encoder output?
#' @param ... Not used.
#' @return The latent components, one column per component.
#' @seealso [predict.nlbss()]
#' @examplesIf tf_available()
#' x <- ts(mix_data(matrix(rnorm(3000), ncol = 3), n_layers = 2))
#' fit <- VAE(x, epochs = 2, verbose = 0)
#' class(components(fit))
#' @exportS3Method generics::components
components.nlbss <- function(object, format = c("input", "matrix"),
                             unscaled = FALSE, ...) {
  M <- if (unscaled) object$S_unscaled else object$S
  restore_format(object$data, M, match.arg(format))
}

#' @importFrom generics components
#' @export
generics::components

#' Reconstructed Data
#'
#' Returns the training data encoded and decoded by the fitted model, in the
#' format of the input data.
#'
#' @param object A fitted model from [iVAE()], [iVAEar()] or [VAE()].
#' @inheritParams components.nlbss
#' @param ... Not used.
#' @return The reconstructed data.
#' @export
fitted.nlbss <- function(object, format = c("input", "matrix"), ...) {
  if (object$method == "TCL") stop("TCL models have no decoder.", call. = FALSE)
  restore_format(object$data, decode_unscaled(object, object$S_unscaled),
                 match.arg(format))
}

#' @export
print.nlbss <- function(x, ...) {
  cat(method_title(x$method), "\n\n", sep = "")
  cat("Call: ", paste(deparse(x$call), collapse = "\n"), "\n", sep = "")
  cat("Data: ", x$data_name, " (", x$n, " observations, ", x$p,
      " variables, format ", x$data$format, ")\n", sep = "")
  cat("Latent components: ", x$latent_dim, if (x$ar_order > 0)
    paste0(", autoregressive order ", x$ar_order), "\n", sep = "")
  if (x$method == "GiVAE") {
    for (g in x$graphs) cat("  ", format_graph(g), "\n", sep = "")
    cat("  Neighbour slots per graph: ", x$neighbors$M, ", rho_max: ", x$rho_max, "\n", sep = "")
    if (!is.null(x$jacobian) && x$jacobian$mode != "none") {
      cat("  Jacobian term: ", x$jacobian$mode, " (", x$jacobian$components,
          " components, largest ", x$jacobian$largest, ")\n", sep = "")
    } else {
      cat("  Jacobian term: none\n")
    }
  }
  if (x$method %in% c("iVAE", "iVAEar", "GiVAE")) {
    for (s in x$aux$specs) cat(format(s), "\n", sep = "")
    if (isTRUE(x$aux$batch)) cat("Auxiliary variables computed batch-wise\n")
  }
  if (x$method == "TCL") {
    cat("Segments: ", x$spec$n_classes, ", training accuracy: ",
        format(x$accuracy, digits = 3), "\n", sep = "")
  } else {
    cat("Distributions: source ", x$spec$source_dist, ", error ",
        x$spec$error_dist, " (sigma ",
        paste(x$spec$error_dist_sigma, collapse = " -> "), ")\n", sep = "")
    cat("ELBO:", format(x$elbo, digits = 5), "\n")
  }
  invisible(x)
}

#' Summary of a Nonlinear BSS Model
#'
#' Summarises the fitted model, the training and the estimated components.
#'
#' @param object A fitted model from [iVAE()], [iVAEar()], [VAE()] or [TCL()].
#' @param ... Not used.
#' @return An object of class `summary.nlbss`, a list with the model details,
#'   final training metrics, the correlations of the components and, for
#'   iVAE type models, summaries of the prior variances and autoregressive
#'   coefficients over the training data.
#' @export
summary.nlbss <- function(object, ...) {
  h <- object$history
  last <- function(metric, data = "training") {
    v <- h$value[h$metric == metric & h$data == data]
    if (length(v)) v[length(v)] else NA_real_
  }
  out <- list(
    method = object$method, call = object$call, data_name = object$data_name,
    n = object$n, p = object$p, format = object$data$format,
    latent_dim = object$latent_dim, ar_order = object$ar_order,
    aux = if (!is.null(object$aux)) vapply(object$aux$specs, format, ""),
    spec = object$spec, elbo = object$elbo,
    epochs = max(h$epoch),
    loss = c(training = last("loss"), validation = last("loss", "validation")),
    max_cor = max(abs(cor(object$S)[upper.tri(diag(object$latent_dim))]),
                  0, na.rm = TRUE),
    accuracy = object$accuracy
  )
  if (!is.null(object$prior)) {
    v <- prior_variance(object, object$prior$log_var)
    v <- sweep(v, 2, object$S_scale^2, "/")
    out$variance <- t(apply(v, 2, stats::quantile, probs = c(0, 0.5, 1)))
    rownames(out$variance) <- paste0("IC", seq_len(object$latent_dim))
    if (object$ar_order > 0) {
      out$ar <- do.call(rbind, lapply(seq_len(object$ar_order), function(r) {
        a <- object$prior$ar[[r]]
        data.frame(component = paste0("IC", seq_len(ncol(a))), lag = r,
                   min = apply(a, 2, min), mean = colMeans(a),
                   max = apply(a, 2, max))
      }))
    }
  }
  class(out) <- "summary.nlbss"
  out
}

#' @export
print.summary.nlbss <- function(x, digits = 3, ...) {
  cat(method_title(x$method), "\n\n", sep = "")
  cat("Call: ", paste(deparse(x$call), collapse = "\n"), "\n\n", sep = "")
  cat("Data:", x$data_name, "\n")
  cat("  observations:", x$n, "\n  variables:", x$p, "\n  format:", x$format, "\n")
  cat("Latent components:", x$latent_dim, "\n")
  if (x$ar_order > 0) cat("Autoregressive order:", x$ar_order, "\n")
  if (!is.null(x$aux)) cat(x$aux, sep = "\n")
  cat("\nNetwork: hidden units ", paste(x$spec$hidden_units, collapse = "-"),
      if (length(x$spec$aux_hidden_units)) paste0(", auxiliary function ",
        paste(x$spec$aux_hidden_units, collapse = "-")),
      ", activation ", x$spec$activation, "\n", sep = "")
  if (x$method != "TCL") {
    cat("Distributions: source ", x$spec$source_dist, ", error ",
        x$spec$error_dist, "\n", sep = "")
  }
  cat("\nTraining: ", x$epochs, " epochs, final loss ",
      format(x$loss["training"], digits = digits), sep = "")
  if (!is.na(x$loss["validation"])) {
    cat(", validation loss", format(x$loss["validation"], digits = digits))
  }
  cat("\n")
  if (x$method == "TCL") {
    cat("Classification accuracy:", format(x$accuracy, digits = digits), "\n")
  } else {
    cat("ELBO:", format(x$elbo, digits = 5), "\n")
  }
  cat("Largest absolute correlation between components:",
      format(x$max_cor, digits = digits), "\n")
  if (!is.null(x$variance)) {
    cat("\nPrior variance of the standardised components over the data:\n")
    print(x$variance, digits = digits)
  }
  if (!is.null(x$ar)) {
    cat("\nAutoregressive coefficients over the data:\n")
    print(x$ar, digits = digits, row.names = FALSE)
  }
  invisible(x)
}

#' Plot a Nonlinear BSS Model
#'
#' Plots the estimated latent components or the training history with
#' ggplot2.
#'
#' @details
#' `type = "series"` plots the components against time (or the observation
#' index); for data with several series (locations or groups), the series
#' given by `series` is shown. `type = "map"` shows the components at the
#' spatial locations, at the time point given by `time` for spatio-temporal
#' data. Polygon data (an `sf` object with areal units, or `STFDF` data with
#' polygons) are drawn as a choropleth, point data as coloured points.
#' `type = "history"` plots the loss for each epoch. The default `"auto"`
#' uses a map for spatial data and series otherwise.
#'
#' @param x A fitted model from [iVAE()], [iVAEar()], [VAE()] or [TCL()].
#' @param type The type of plot, see Details.
#' @param which Indices of the components to plot.
#' @param time For maps of spatio-temporal data, the time point to show.
#'   Defaults to the last time point.
#' @param series For series plots of data with several series, the series
#'   (location index or group) to show. Defaults to the first one.
#' @param unscaled Logical. Plot the unstandardised components?
#' @param ... Further arguments passed to the main geom of the plot
#'   ([ggplot2::geom_line()], [ggplot2::geom_point()] or [ggplot2::geom_sf()]).
#' @return A `ggplot` object, which can be modified with the usual ggplot2
#'   functions.
#' @export
plot.nlbss <- function(x, type = c("auto", "series", "map", "history"),
                       which = seq_len(x$latent_dim), time = NULL,
                       series = NULL, unscaled = FALSE, ...) {
  type <- match.arg(type)
  if (type == "auto") type <- if (!is.null(x$data$coords)) "map" else "series"
  if (type == "history") return(plot_history(x, ...))
  IC <- if (unscaled) x$S_unscaled else x$S
  IC <- IC[, which, drop = FALSE]
  if (type == "map") {
    if (is.null(x$data$coords)) stop("The data have no coordinates.", call. = FALSE)
    plot_map(x, IC, time, ...)
  } else {
    plot_series(x, IC, series, ...)
  }
}

plot_series <- function(x, IC, series, ...) {
  key <- series_key(x$data)
  keys <- unique(key)
  if (is.null(series)) series <- keys[1]
  if (is.numeric(series)) series <- keys[series]
  rows <- which(key == series)
  t <- if (is.null(x$data$time_raw)) rows else x$data$time_raw[rows]
  o <- order(t)
  d <- data.frame(time = rep(t[o], ncol(IC)),
                  value = as.vector(IC[rows[o], , drop = FALSE]),
                  component = factor(rep(colnames(IC), each = length(rows)),
                                     levels = colnames(IC)))
  ggplot2::ggplot(d, ggplot2::aes(.data$time, .data$value)) +
    ggplot2::geom_line(colour = "#2c7fb8", ...) +
    ggplot2::facet_wrap(ggplot2::vars(.data$component), ncol = 1,
                        scales = "free_y", strip.position = "left") +
    ggplot2::labs(x = if (is.null(x$data$time_raw)) "Index" else "Time",
                  y = NULL) +
    theme_nlbss() +
    ggplot2::theme(strip.placement = "outside")
}

plot_map <- function(x, IC, time, ...) {
  rows <- seq_len(nrow(IC))
  if (!is.null(x$data$time)) {
    # a time given by the user is in the units of the original time points
    rows <- if (is.null(time)) {
      which(x$data$time == max(x$data$time))
    } else {
      raw <- if (is.null(x$data$time_raw)) x$data$time else x$data$time_raw
      which(time_to_numeric(raw) == time_to_numeric(time))
    }
    if (length(rows) == 0) stop("No observations at the given time point.", call. = FALSE)
  }
  geometry <- row_geometry(x$data, rows)
  map_plot(IC[rows, , drop = FALSE], geometry = geometry,
           xy = x$data$coords[rows, 1:2, drop = FALSE], fill = "value", ...)
}

plot_history <- function(x, ...) {
  h <- x$history[x$history$metric == "loss", ]
  h$data <- factor(h$data, levels = unique(h$data))
  p <- ggplot2::ggplot(h, ggplot2::aes(.data$epoch, .data$value,
                                       linetype = .data$data)) +
    ggplot2::geom_line(colour = "#2c7fb8", ...) +
    ggplot2::labs(x = "Epoch", y = "Loss", linetype = NULL) +
    theme_nlbss()
  if (nlevels(h$data) < 2L) p <- p + ggplot2::guides(linetype = "none")
  p
}

format_graph <- function(g) {
  deg <- lengths(g$adj_list)
  paste0("Graph", if (!is.null(g$name)) paste0(" '", g$name, "'"), ": ", g$n_nodes,
         " nodes, mean degree ", format(mean(deg), digits = 3),
         if (is.null(g$node)) paste0(", each = ", g$each) else ", explicit layout")
}