#' Absolute Mean Correlation Coefficient
#' @description Calculates absolute mean correlation coefficient
#' \loadmathjax
#' @import RcppHungarian
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
    max_cors <- HungarianSolver(-abs(cor_mat))$cost
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
