#' Absolute Mean Correlation Coefficient
#' @description Calculates absolute mean correlation coefficient
#' \loadmathjax
#' @param cor_mat A correlation matrix.
#' @return
#' A numeric value of absoulte mean correlation coeffiecient
#' @details
#' The method calculates absolute mean correlation coeffiecient (MCC) by
#' solving a linear assignment problem. MCC is mathematically given as
#' \mjdeqn{
#' MCC(\mathbf K)=\frac{1}{p} \sup_{\mathbf P  \in \mathcal{P}}
#' tr(\mathbf P\, \abs(\mathbf K)),
#' }{ascii}
#' where \mjeqn{\mathcal{P}}{ascii} is a set of all possible
#' permutation matrices,
#' \mjeqn{ tr(\cdot )}{ascii} is the trace of a matrix, and
#' \mjeqn{\abs(\cdot)}{ascii}
#' denotes taking the absolute value of a matrix elementwise.
#' @examples
#' d1 <- matrix(rnorm(300), ncol = 3)
#' d2 <- d1 + matrix(rnorm(300), ncol = 3)
#' cor_mat <- cor(d1, d2)
#' absolute_mean_correlation(cor_mat)
#' @export
absolute_mean_correlation <- function(cor_mat) {
    max_cors <- RcppHungarian::HungarianSolver(-abs(cor_mat))$cost
    -max_cors / dim(cor_mat)[1]
}

l2normalize <- function(A) {
    svd(A)$u
}

lrelu <- function(a, slope = 0.2) {
    if (a < 0) {
        a <- a * slope
    }
    return(a)
}

invlrelu <- function(a) {
    if (a < 0) {
        a <- a * 10
    }
    return(a)
}

xtanh <- function(a, mult = 1, slope = 0.1) {
    tanh(mult * a) + slope * a
}

elu <- function(a) {
    if (a < 0) {
        a <- exp(a) - 1
    } else {
        a <- a
    }
    return(a)
}

exp_arctan <- function(a) {
    (1 / 2) * a * exp(atan(a))
}

leaky_softplus <- function(a, slope = 0.1) {
    slope * a + (1 - slope) * log(1 + exp(a))
}

logsumexp <- function(a, slope = 0.1) {
    log(exp(slope * a) + exp(a)) - log(2)
}

smooth_leaky_relu <- function(a, alpha = 0.05, eps = 0.1) {
    a <- 0.5 * ((1 + alpha) * a + (1 - alpha) * sqrt(a^2 + eps))
    return(a)
}

gaussian_kernel <- function(d) {
    exp(-d^2)
}

wendland_kernel <- function(d) {
    ((1 - d)^6 * (35 * d^2 + 18 * d + 3) / 3) * ifelse(d >= 1, 0, 1)
}

#' Create Overdeterminated Mixed Data
#' @description Creates overdeterminated mixed data based on the input data.
#' \loadmathjax
#' @param data A matrix containing the original data.
#' @param dim_mixed The dimension of the mixed data.
#' @param n_layers The number of mixing layers.
#' @param nonlinearity The function for applying nonlinearity
#' between the mixing layers.
#' @return
#' A matrix containing the mixed data.
#' @details
#' The method mixes the input data by applying mixing matrices and nonlinear
#' activation function based on the input parameters.
#' Let \mjeqn{\omega_i}{ascii} be
#' the activation function of \mjeqn{i}{ascii}th layer and
#' \mjeqn{\mathbf B_i}{ascii} be the
#' normalized mixing matrix of \mjeqn{i}{ascii}th layer.
#' Then, the mixing function \mjeqn{\mathbf f_L}{ascii}
#' is defined as
#' \mjdeqn{
#'     \mathbf f_L(\mathbf z) = \begin{cases}
#'         \omega_L(\mathbf B_L \mathbf z),\quad L = 1, \\
#'         \omega_L(\mathbf B_L \mathbf f_{L-1}(\mathbf z)),
#'         \quad L \in \{2,3,\dots\}.
#'     \end{cases}
#' }{ascii}
#' The function has a linear activation
#' \mjeqn{\omega_L(x)=x}{ascii} for the last layer,
#' which means that mixing function
#' \mjeqn{\mathbf f_1}{ascii} with one layer corresponds
#' to a linear mixing. If more than one layers are applied, the other layers
#' use the activation function given by the parameter \code{nonlinearity}.
#' The options for \code{nonlinearity} are:
#'
#' \code{nonlinarity="elu"}:
#' \mjdeqn{
#' \omega_i(x)=\begin{cases}
#'     x,\quad x \geq 0, \\
#'     exp(x) - 1,\quad x < 0,
#' \end{cases}
#' }{ascii}
#'
#' \code{nonlinarity="xtanh"}:
#' \mjdeqn{
#' \omega_i(x)= tanh(x) + 0.1 * x
#' }{ascii}
#'
#' \code{nonlinarity="lrelu"}:
#' \mjdeqn{
#' \omega_i(x)=\begin{cases}
#'     x,\quad x \geq 0, \\
#'     0.2 * x, x < 0.
#' \end{cases}
#' }{ascii}
#' @examples
#' data <- matrix(rnorm(300), ncol = 3)
#' mixed_data <- mix_data_over_determinated(data, 5, 3)
#' @export
mix_data_over_determinated <- function(
    data, dim_mixed, n_layers = 1,
    nonlinearity = "elu") {
    data_dim <- dim(data)[2]
    A <- matrix(runif(data_dim * dim_mixed, -1, 1), ncol = data_dim)
    A <- l2normalize(A)
    mixed_data <- t(A %*% t(data))
    mixed_data <- mixed_data / mean(sqrt(diag(var(mixed_data))))
    if (n_layers > 1) {
        if (nonlinearity == "xtanh") {
            mixed_data <- apply(mixed_data, c(1, 2), xtanh)
        } else if (nonlinearity == "elu") {
            mixed_data <- apply(mixed_data, c(1, 2), elu)
        } else if (nonlinearity == "smooth_lrelu") {
            mixed_data <- apply(mixed_data, c(1, 2), smooth_leaky_relu)
        } else {
            mixed_data <- apply(mixed_data, c(1, 2), lrelu)
        }
    }
    if (n_layers == 1) {
        return(mixed_data)
    } else {
        mix_data_over_determinated(
            mixed_data, dim_mixed, n_layers - 1,
            nonlinearity
        )
    }
}

#' Create Mixed Data
#' @description Creates mixed data based on the input data.
#' \loadmathjax
#' @param data A matrix containing the original data.
#' @param n_layers The number of mixing layers.
#' @param nonlinearity The function for applying nonlinearity
#' between the mixing layers.
#' @return
#' A matrix containing the mixed data.
#' @details
#' The method mixes the input data by applying mixing matrices and nonlinear
#' activation function based on the input parameters.
#' Let \mjeqn{\omega_i}{ascii} be
#' the activation function of
#' \mjeqn{i}{ascii}th layer and \mjeqn{\mathbf B_i}{ascii} be the
#' normalized mixing matrix of \mjeqn{i}{ascii}th layer.
#' Then, the mixing function \mjeqn{\mathbf f_L}{ascii} is defined as
#' \mjdeqn{
#'     \mathbf f_L(\mathbf z) = \begin{cases}
#'         \omega_L(\mathbf B_L \mathbf z),\quad L = 1, \\
#'         \omega_L(\mathbf B_L \mathbf f_{L-1}(\mathbf z)),
#'         \quad L \in \{2,3,\dots\}.
#'     \end{cases}
#' }{ascii}
#' The function has a linear activation \mjeqn{\omega_L(x)=x}{ascii} for the last layer,
#' which means that mixing function \mjeqn{\mathbf f_1}{ascii} with one layer corresponds
#' to a linear mixing. If more than one layers are applied, the other layers
#' use the activation function given by the parameter \code{nonlinearity}.
#' The options for \code{nonlinearity} are:
#'
#' \code{nonlinarity="elu"}:
#' \mjdeqn{
#' \omega_i(x)=\begin{cases}
#'     x,\quad x \geq 0, \\
#'     exp(x) - 1,\quad x < 0,
#' \end{cases}
#' }{ascii}
#'
#' \code{nonlinarity="xtanh"}:
#' \mjdeqn{
#' \omega_i(x)= tanh(x) + 0.1 * x
#' }{ascii}
#'
#' \code{nonlinarity="lrelu"}:
#' \mjdeqn{
#' \omega_i(x)=\begin{cases}
#'     x,\quad x \geq 0, \\
#'     0.2 * x, x < 0.
#' \end{cases}
#' }{ascii}
#' @examples
#' data <- matrix(rnorm(300), ncol = 3)
#' mixed_data <- mix_data(data, 3)
#' @export
mix_data <- function(data, n_layers = 1, nonlinearity = "elu") {
    hidden_dim <- dim(data)[2]
    mixed_data <- data
    if (n_layers > 1) {
        for (i in 2:n_layers) {
            A <- matrix(runif(hidden_dim^2, -1, 1), ncol = hidden_dim)
            A <- l2normalize(A)
            mixed_data <- t(A %*% t(mixed_data))
            mixed_data <- mixed_data / mean(sqrt(diag(var(mixed_data))))
            if (nonlinearity == "xtanh") {
                mixed_data <- apply(mixed_data, c(1, 2), xtanh)
            } else if (nonlinearity == "elu") {
                mixed_data <- apply(mixed_data, c(1, 2), elu)
            } else if (nonlinearity == "exp_arctan") {
                mixed_data <- apply(mixed_data, c(1, 2), exp_arctan)
            } else if (nonlinearity == "leaky_softplus") {
                mixed_data <- apply(mixed_data, c(1, 2), leaky_softplus)
            } else if (nonlinearity == "logsumexp") {
                mixed_data <- apply(mixed_data, c(1, 2), logsumexp)
            } else if (nonlinearity == "smooth_lrelu") {
                mixed_data <- apply(mixed_data, c(1, 2), smooth_leaky_relu)
            } else {
                mixed_data <- apply(mixed_data, c(1, 2), lrelu)
            }
        }
    }
    A <- matrix(runif(hidden_dim^2, -1, 1), ncol = hidden_dim)
    A <- l2normalize(A)
    mixed_data <- t(A %*% t(mixed_data))
    mixed_data <- mixed_data / mean(sqrt(diag(var(mixed_data))))
    return(mixed_data)
}

#' Generate Nonstationary Spatial Data by Segments
#' @description Generates nonstationary Gaussian spatial data with
#' changing mean and variance by segments. \loadmathjax
#' @param n A sample size.
#' @param p A dimension of the data.
#' @param coords A 2 x n matrix containing the spatial coordinates.
#' @param n_segments The number of spatial segments. Each segment
#' has their own mean and variance
#' @param random_mean A boolean determining if the constant zero mean
#' is used or if mean is random randomly sampled from
#' \mjeqn{Unif(-5, 5)}{ascii}
#' for each segment.
#' @return
#' An object with the following properties
#' \item{data}{A p x n matrix containing the generated data.}
#' \item{labels}{A vector giving numerical labels for each data point.
#' The data points with same label belong in the same segment.}
#' \item{coords}{A 2 x n matrix with the spatial coordinates.}
#' \item{center_points}{A vector of length \code{n_segments} containing
#' the center points for each segment.}
#' @details
#' The method chooses uniformly \code{n_segments} center points from the provided
#' spatial locations. The locations are divided into segments based on the
#' lowest distance to any of the center points. The data is generated from
#' normal distribution by sampling unique variance from distribution
#' \mjeqn{Unif(0.1, 5)}{ascii} for each segment. If \code{random_mean = TRUE}
#' the means for each segment are sampled from distribution \mjeqn{Unif(-5, 5)}{ascii}.
#' If \code{random_mean = FALSE} the constant zero mean is used for every segment.
#' @examples
#' coords <- matrix(runif(2000, 0, 1), ncol = 2)
#' data_obj <- generate_nonstationary_spatial_data_by_segments(1000, 3, coords, 10)
#' @export
generate_nonstationary_spatial_data_by_segments <-
    function(n, p, coords, n_segments, random_mean = TRUE) {
        data <- matrix(NA, nrow = n, ncol = p)
        labels <- numeric(n)
        center_point_inds <- sample(1:n, n_segments)
        center_points <- coords[center_point_inds, ]
        labels <- unlist(apply(coords, 1, FUN = function(coord) {
            centered_points <- matrix(unlist(apply(center_points, 1,
                FUN = function(x) {
                    x - coord
                }
            )), byrow = TRUE, nrow = n_segments)
            dists <- apply(centered_points, 1, FUN = function(x) {
                sqrt(x[1]^2 + x[2]^2)
            })
            return(which(dists == min(dists))[1])
        }))
        for (k in 1:n_segments) {
            segment_coord_inds <- which(labels == k)
            segment_coords <- coords[segment_coord_inds, ]
            for (i in 1:p) {
                mean_val <- 0
                if (random_mean) {
                    mean_val <- runif(1, -5, 5)
                }
                data[segment_coord_inds, i] <- rnorm(
                    nrow(segment_coords),
                    mean_val, runif(1, 0.1, 5)
                )
            }
        }
        return(list(
            data = data, labels = labels, coords = coords,
            center_points = center_points
        ))
    }

#' Generate Nonstationary Spatio-Temporal Data by Segments
#' @description Generates nonstationary Gaussian spatio-temporal data with
#' changing mean and variance by segments. \loadmathjax
#' @param n_time Number of time points.
#' @param n_spat Number of spatial points.
#' @param p A dimension of the data.
#' @param n_segments_space Number of spatial segments.
#' @param n_segments_time Number of temporal segments.
#' @param coords_time A 3 x n (x, y, time) matrix containing the 
#' spatio-temporal coordinates.
#' @return
#' An object with the following properties
#' \item{data}{A p x n matrix containing the generated data.}
#' \item{labels}{A vector giving numerical labels for each data point.
#' The data points with same label belong in the same segment.}
#' \item{coords}{A 3 x n matrix containing the spatio-temporal coordinates.}
#' @details
#' The method chooses uniformly \code{n_segments_space} center points from 
#' the provided spatial locations. The locations are divided into segments 
#' based on the lowest distance to any of the center points. Similarly 
#' \code{n_segments_time} center points are selected from the provided 
#' temporal points and the data are divided into temporal segments.
#' The data is generated from normal distribution by sampling unique variance 
#' from distribution \mjeqn{Unif(0.1, 5)}{ascii} for each segment and mean from
#' distribution \mjeqn{Unif(-1, 1)}{ascii}.
#' @examples
#' coords_time <- cbind(rep(runif(50), 100), rep(runif(50), 100), rep(1:100, each=50))
#' data_obj <- generate_nonstationary_spatio_temporal_data_by_segments(100, 50, 3, 5, 10, coords_time)
#' @export
generate_nonstationary_spatio_temporal_data_by_segments <- 
    function(n_time, n_spat, p, n_segments_space, n_segments_time, coords_time) {
  n <- dim(coords_time)[1]
  data <- matrix(NA, nrow = n, ncol = p)
  labels <- numeric(n)
  center_point_inds <- sample(1:n_spat, n_segments_space)
  center_points <- coords_time[center_point_inds,1:2]
  labels_spat <- unlist(apply(coords_time[,1:2], 1, FUN = function(coord) {
    centered_points <- matrix(unlist(apply(center_points, 1, FUN = function(x) { x - coord })), byrow = TRUE, nrow = n_segments_space)
    dists <- apply(centered_points, 1, FUN = function(x) {sqrt(x[1]^2 + x[2]^2)})
    return(which(dists == min(dists))[1])
  }))
  center_times <- sample(1:n_time, n_segments_time)
  labels_time <- unlist(sapply(coords_time[,3], FUN = function(t) {
    dists <- matrix(unlist(sapply(center_times, FUN = function(x) { abs(x - t) })), byrow = TRUE, nrow = n_segments_time)
    return(which(dists == min(dists))[1])
  }))
  for (k in 1:n_segments_space) {
    for (k2 in 1:n_segments_time) {
      segment_coord_inds <- intersect(which(labels_spat == k), which(labels_time == k2))
      segment_coords <- coords_time[segment_coord_inds, ]
      for (i in 1:p) {
        data[segment_coord_inds, i] <- rnorm(nrow(segment_coords), runif(1, -1 ,1), runif(1,0.1,5))
      } 
    }
  }
  return(list(data = data, labels = labels, coords_time = coords_time))
}

mix_data_acyclic <- function(
    data, dim_mixed, n_layers = 1,
    nonlinearity = "elu") {
    data_dim <- dim(data)[2]
    edge_matrices <- list()
    mixed_data <- data
    edges <- ifelse(runif(dim_mixed * dim_mixed) > (2 / (dim_mixed - 1)), 0, 1)
    parent_total_weights <- runif(dim_mixed, 0.2, 3)
    edge_value_array <- array(runif(dim_mixed * dim_mixed * n_layers, -1, 1),
                              dim = c(dim_mixed, dim_mixed, n_layers))
    for (i in 1:n_layers) {
        edge_value_array[, , i] <- (edge_value_array[, , i] * edges) * lower.tri(edge_value_array[, , i])
    }
    #edge_value_array_scales <- apply(edge_value_array, 1, sum)
    #edge_value_array_scales[edge_value_array_scales == 0] <- 1
    #rescale_values <- parent_total_weights / abs(edge_value_array_scales)
    #for (i in 1:dim_mixed) {
    #    edge_value_array[i, , ] <- edge_value_array[i, , ] * rescale_values[i]
    #}
    for (i in 1:n_layers) {
        A <- diag(dim_mixed)
        edge_matrices <- append(edge_matrices, list(edge_value_array[, , i]))
        A <- A + edge_value_array[, , i]
        #A <- l2normalize(A)
        mixed_data <- t(A %*% t(mixed_data))
        mixed_data <- mixed_data / mean(sqrt(diag(var(mixed_data))))
        if (i > 1) {
            if (nonlinearity == "xtanh") {
                mixed_data <- apply(mixed_data, c(1, 2), xtanh)
            } else if (nonlinearity == "elu") {
                mixed_data <- apply(mixed_data, c(1, 2), elu)
            } else {
                mixed_data <- apply(mixed_data, c(1, 2), lrelu)
            }
        }
    }
    return(list(data = mixed_data, edge_matrices = edge_matrices))
}
