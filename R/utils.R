norm_log_pdf <- function(x, mu, v, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    lpdf <- tensorflow::tf$constant(-0.5, "float32") * (tensorflow::tf$math$pow((x - mu), 2) / v + tensorflow::tf$cast(tensorflow::tf$math$log(2 * pi), "float32") + tensorflow::tf$math$log(v))
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

huber_loss <- function(x, mu, v, delta = 0.2, reduce = TRUE) {
    v <- v + 1e-35 # To avoid computational problems (when v == 0)
    abs_diff <- tensorflow::tf$abs(x - mu)
    delta <- tensorflow::tf$constant(delta, "float32")
    quadratic_part <- tensorflow::tf$constant(-1/2, "float32") * tensorflow::tf$square(x - mu)
    linear_part <- -delta * (abs_diff - tensorflow::tf$constant(0.5, "float32") * delta)
    huber_loss <- tensorflow::tf$where(abs_diff < delta, quadratic_part, linear_part)
    lpdf <- huber_loss / v # + tensorflow::tf$constant(-0.5, "float32") * (tensorflow::tf$cast(tensorflow::tf$math$log(2 * pi), "float32") + tensorflow::tf$math$log(v))
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

laplace_log_pdf <- function(x, loc, scale, reduce = TRUE) {
    scale <- scale + 1e-35
    lpdf <- -(tensorflow::tf$math$log(2 * scale) + tensorflow::tf$abs(x - loc) / (scale))
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

# sigma not used
bernoulli_log_pdf <- function(x, theta, sigma, reduce = TRUE) {
    theta <- theta * 0.99 + 0.005
    lpdf <- x * tensorflow::tf$math$log(theta) + (1 - x) * tensorflow::tf$math$log(1 - theta)
    if (reduce) {
        return(tensorflow::tf$reduce_sum(lpdf, -1L))
    }
    return(lpdf)
}

#' Poisson Log-Likelihood
#' @description Log-likelihood of x under Poisson(exp(log_rate)).
#' @param x Observed counts (non-negative integers).
#' @param log_rate Log of the Poisson rate parameter (raw decoder output).
#' @param v Ignored; present for interface compatibility with norm_log_pdf.
#' @param reduce If TRUE, sum over the feature dimension.
poisson_log_pdf <- function(x, log_rate, v = NULL, reduce = TRUE) {
  x_f  <- tensorflow::tf$cast(x, "float32")
  # log p(x | lambda) = x * log(lambda) - lambda  (omitting log(x!) as constant)
  lpdf <- x_f * log_rate - tensorflow::tf$math$exp(log_rate)
  if (reduce) return(tensorflow::tf$reduce_sum(lpdf, -1L))
  return(lpdf)
}

absolute_activation <- keras3::Layer(
    "AbsoluteActivation",
    initialize = function() {
        super$initialize()
    },
    call = function(y) {
        tensorflow::tf$abs(y)
    }
)

sampling_gaussian <- keras3::Layer(
    "SamplingGaussian",
    initialize = function(p) {
        super$initialize()
        self$p <- p
    },
    call = function(y) {
        z_mean <- y[, 1:self$p]
        z_log_var <- y[, (self$p + 1):(2 * self$p)]
        epsilon <- tensorflow::tf$random$normal(shape(self$p))
        z_mean + tensorflow::tf$math$exp(z_log_var / 2) * epsilon
    }
)

sampling_laplace <- keras3::Layer(
    "SamplingLaplace",
    initialize = function(p) {
        super$initialize()
        self$p <- p
    },
    call = function(y) {
        z_mean <- y[, 1:self$p]
        log_scale <- y[, (self$p + 1):(2 * self$p)]
        cum_vals <- tensorflow::tf$random$uniform(shape(self$p))
        epsilon <- tensorflow::tf$sign(cum_vals - 0.5) * log(2 * tensorflow::tf$abs(cum_vals - 0.5))
        z_mean - tensorflow::tf$math$exp(log_scale) * epsilon
    }
)

find_unique_name <- function(name, i = 0) {
    if (dir.exists(name)) {
        find_unique_name(paste0(name, i + 1))
    } else {
        return(name)
    }
}

compute_laplacian_eigenvectors <- function(adj_list, K = 10) {
    # adj_list: list of length n, each element is integer vector of neighbor indices (1-based)
    n <- length(adj_list)
    
    # Build sparse adjacency and degree matrices
    i_idx <- c(); j_idx <- c()
    for (i in seq_len(n)) {
        for (j in adj_list[[i]]) {
            i_idx <- c(i_idx, i)
            j_idx <- c(j_idx, j)
        }
    }
    W <- Matrix::sparseMatrix(i_idx, j_idx, x = 1, dims = c(n, n))
    D <- Matrix::Diagonal(x = rowSums(as.matrix(W)))
    L <- D - W  # Graph Laplacian
    
    # Compute K smallest eigenvectors
    # Skip the trivial k=1 (constant, eigenvalue=0), return k=2..K+1
    # Using RSpectra for large graphs, base eigen() for small ones
    if (n <= 500) {
        eig <- eigen(as.matrix(L), symmetric = TRUE)
        # eigen() returns in DECREASING order, so smallest are at the end
        idx <- (n - K):n  # includes trivial eigenvector at end
        vecs <- eig$vectors[, rev(idx)]  # now increasing eigenvalue order
        vals <- rev(eig$values[idx])
    } else {
        # RSpectra is much faster for large sparse matrices
        # finds K+1 smallest eigenvalues
        eig <- RSpectra::eigs_sym(L, k = K + 1, which = "SM")
        ord <- order(eig$values)
        vecs <- eig$vectors[, ord]
        vals <- eig$values[ord]
    }
    
    # Column 1 is trivial (eigenvalue ~0), columns 2..K+1 are the useful ones
    phi <- vecs[, 2:(K + 1), drop = FALSE]
    lambda <- vals[2:(K + 1)]
    
    # Fix sign convention: make largest-magnitude entry positive in each vector
    for (k in seq_len(ncol(phi))) {
        if (phi[which.max(abs(phi[, k])), k] < 0) phi[, k] <- -phi[, k]
    }
    
    colnames(phi) <- paste0("phi_", seq_len(K))
    return(list(phi = phi, lambda = lambda, L = L))
}