 ------------------------------------------------------------------------------
# Internal helper: pre-build padded neighbor data matrices from adj_list
# Returns one matrix per neighbor slot k (1 .. max_n_adj), where row i holds
# the data/aux of the k-th neighbor of observation i (zeros if no k-th neighbor).
# This mirrors how iVAEar constructs prev_data_list for AR lags.
# ------------------------------------------------------------------------------
 
.build_neighbor_matrices <- function(data_scaled, aux_data, adj_list) {
  n         <- nrow(data_scaled)
  p         <- ncol(data_scaled)
  K         <- ncol(aux_data)
  max_n_adj <- max(sapply(adj_list, length))
 
  neigh_data_list <- vector("list", max_n_adj)
  neigh_aux_list  <- vector("list", max_n_adj)
  # neigh_valid_mat[i, k] = 1 iff observation i has at least k neighbors
  neigh_valid_mat <- matrix(0L, nrow = n, ncol = max_n_adj)
 
  for (k in seq_len(max_n_adj)) {
    nd <- matrix(0.0, nrow = n, ncol = p)
    na <- matrix(0.0, nrow = n, ncol = K)
    for (i in seq_len(n)) {
      if (length(adj_list[[i]]) >= k) {
        j              <- adj_list[[i]][k]
        nd[i, ]        <- data_scaled[j, ]
        na[i, ]        <- aux_data[j, ]
        neigh_valid_mat[i, k] <- 1L
      }
    }
    neigh_data_list[[k]] <- nd
    neigh_aux_list[[k]]  <- na
  }
 
  list(
    neigh_data_list = neigh_data_list,
    neigh_aux_list  = neigh_aux_list,
    neigh_valid_mat = neigh_valid_mat,
    max_n_adj       = as.integer(max_n_adj)
  )
}


# ------------------------------------------------------------------------------
# GiVAE base function
# ------------------------------------------------------------------------------
 
#' Graph-structured Identifiable Variational Autoencoder
#'
#' @description
#' Trains an iVAE whose latent distribution is assumed to follow
#' a nonstationary spatial SAR model.
#' For each observation i the prior mean is
#'   \eqn{\mu_i = \rho(u_i) \sum_{j \in \partial i} w_{ij}\, \hat{z}_j}
#' where \eqn{\hat{z}_j = \text{encoder}(x_j, u_j)} and \eqn{\rho(u_i)} is a
#' location-specific coupling strength estimated from auxiliary data \eqn{u_i}.
#' The prior variance \eqn{\sigma^2(u_i)} is also nonstationary, providing the
#' identifiability leverage of standard iVAE. Edge weights \eqn{w_{ij}} are
#' either uniform (1/|N(i)|) or learned via a bilinear attention model on
#' \eqn{(u_i, u_j)}.
#'
#' @param data       N x P matrix of observed data.
#' @param aux_data   N x K matrix of auxiliary data (e.g. Laplacian eigenvectors).
#' @param latent_dim Integer number of latent components.
#' @param adj_list   List of length N; each element is a 1-based integer vector
#'   of neighbor indices for that observation.
#' @param learn_edge_weights Logical. If TRUE, learn an MLP-based attention model
#'   for asymmetric edge weights using \eqn{(u_i, u_j)}. Default FALSE.
#' @param rho_max Numeric in (0,1). Hard upper bound on the spatial coupling
#'   parameter \eqn{\rho}. Default 0.9.
#' @param hidden_units      Integer vector; hidden layer sizes for encoder/decoder.
#' @param aux_hidden_units  Integer vector; hidden layer sizes for prior-variance
#'   and rho networks.
#' @param edge_hidden_units Integer vector; hidden layer sizes for edge weight
#'   model (used only when \code{learn_edge_weights = TRUE}).
#' @param activation Activation function string. Default "leaky_relu".
#' @param source_dist Latent distribution: "gaussian" or "laplace".
#' @param error_dist  Reconstruction distribution: "gaussian", "laplace", or
#'   "poisson". For "poisson", the decoder outputs log-rates and data is not
#'   standardised.
#' @param error_dist_sigma Scale parameter for gaussian/laplace error (ignored
#'   for poisson). Default 0.02.
#' @param optimizer  Optional Keras optimizer; default is Adam with polynomial
#'   learning-rate decay.
#' @param lr_start,lr_end,steps Learning-rate schedule parameters.
#' @param seed Optional integer seed.
#' @param epochs,batch_size Training parameters.
#' @param validation_split Fraction of data reserved for validation.
#'
#' @return An object of class \code{"GiVAE"}.
#' @export
GiVAE <- function(
    data,
    aux_data,
    latent_dim,
    adj_list,
    learn_edge_weights = FALSE,
    rho_max            = 0.9,
    hidden_units       = c(128, 128, 128),
    aux_hidden_units   = c(128, 128, 128),
    edge_hidden_units  = c(64, 64),
    activation         = "leaky_relu",
    source_dist        = "gaussian",
    error_dist         = "gaussian",
    error_dist_sigma   = 0.02,
    optimizer          = NULL,
    lr_start           = 0.001,
    lr_end             = 0.0001,
    steps              = 10000,
    seed               = NULL,
    epochs,
    batch_size,
    validation_split   = 0
) {
  source_dist <- match.arg(source_dist, c("gaussian", "laplace"))
  source_log_pdf <- switch(source_dist,
    "gaussian" = norm_log_pdf,
    "laplace"  = laplace_log_pdf
  )
  error_dist <- match.arg(error_dist, c("gaussian", "laplace", "poisson"))
  error_log_pdf <- switch(error_dist,
    "gaussian" = norm_log_pdf,
    "laplace"  = laplace_log_pdf,
    "poisson"  = poisson_log_pdf
  )
 
  call_params <- list(
    latent_dim = latent_dim, source_dist = source_dist,
    error_dist = error_dist, error_dist_sigma = error_dist_sigma,
    hidden_units = hidden_units, aux_hidden_units = aux_hidden_units,
    edge_hidden_units = edge_hidden_units, activation = activation,
    learn_edge_weights = learn_edge_weights, rho_max = rho_max,
    epochs = epochs, batch_size = batch_size,
    lr_start = lr_start, lr_end = lr_end, seed = seed
  )
 
  # ---- Dimensions ------------------------------------------------------------
  n <- as.integer(nrow(data))
  p <- as.integer(ncol(data))
  K <- as.integer(ncol(aux_data))
 
  if (n != nrow(aux_data))   stop("data and aux_data must have the same number of rows")
  if (n != length(adj_list)) stop("length(adj_list) must equal nrow(data)")
 
  # ---- Data scaling ----------------------------------------------------------
  # Poisson: leave as raw counts (decoder outputs log-rate)
  if (error_dist == "poisson") {
    data_means  <- rep(0.0, p)
    data_sds    <- rep(1.0, p)
    data_scaled <- data * 1.0
  } else {
    data_means  <- colMeans(data, na.rm = TRUE)
    data_sds    <- apply(data, 2, sd, na.rm = TRUE)
    data_scaled <- scale(data, center = data_means, scale = data_sds)
  }
 
  # ---- Pre-build neighbor matrices -------------------------------------------
  # For each neighbor slot k = 1..max_n_adj:
  #   neigh_data_list[[k]][i,] = data_scaled[adj_list[[i]][k],]  (or 0)
  #   neigh_aux_list[[k]][i,]  = aux_data[adj_list[[i]][k],]     (or 0)
  #   neigh_valid_mat[i,k]     = 1 iff obs i has >= k neighbors
  # This mirrors iVAEar's construction of prev_data_list for AR lags.
  neigh_obj       <- .build_neighbor_matrices(data_scaled, aux_data, adj_list)
  neigh_data_list <- neigh_obj$neigh_data_list
  neigh_aux_list  <- neigh_obj$neigh_aux_list
  neigh_valid_mat <- neigh_obj$neigh_valid_mat
  max_n_adj       <- neigh_obj$max_n_adj
 
  if (!is.null(seed)) tensorflow::tf$keras$utils$set_random_seed(as.integer(seed))
  latent_dim <- as.integer(latent_dim)
 
  # ---- Model input layers ----------------------------------------------------
  input_data  <- keras3::layer_input(p)           # current observation
  input_aux   <- keras3::layer_input(K)            # current auxiliary
 
  # One pair of inputs per neighbor slot (same pattern as iVAEar's prev inputs)
  neigh_data_inputs <- lapply(seq_len(max_n_adj), function(k) keras3::layer_input(p))
  neigh_aux_inputs  <- lapply(seq_len(max_n_adj), function(k) keras3::layer_input(K))
 
  # Binary validity flags: which neighbor slots are filled for each observation
  input_neigh_valid <- keras3::layer_input(max_n_adj)
 
  # ---- Encoder with shared layers applied to current + all neighbor slots ----
  # Exactly mirrors how iVAEar applies the same dense layers to current and
  # prev_data to produce prev_z encodings.
 
  # Running tensors: enc_current for obs i, neigh_enc[[k]] for k-th neighbor
  enc_current <- keras3::layer_concatenate(list(input_data, input_aux))
  neigh_enc   <- lapply(seq_len(max_n_adj), function(k)
    keras3::layer_concatenate(list(neigh_data_inputs[[k]], neigh_aux_inputs[[k]]))
  )
 
  for (n_units in hidden_units) {
    shared_enc_layer <- keras3::layer_dense(units = n_units, activation = activation)
    enc_current      <- enc_current %>% shared_enc_layer()
    neigh_enc        <- lapply(neigh_enc, function(h) h %>% shared_enc_layer())
  }
 
  z_mean_layer    <- keras3::layer_dense(units = latent_dim)
  z_log_var_layer <- keras3::layer_dense(units = latent_dim)
 
  z_mean         <- enc_current %>% z_mean_layer()      # (batch, latent_dim)
  z_log_var      <- enc_current %>% z_log_var_layer()   # (batch, latent_dim)
  z_mean_and_var <- keras3::layer_concatenate(list(z_mean, z_log_var))

  # Sub-models: encoder = z_mean only (for prediction), z_log_var for ELBO
  encoder         <- keras3::keras_model(list(input_data, input_aux), z_mean)
  z_log_var_model <- keras3::keras_model(list(input_data, input_aux), z_log_var)
 
  # Neighbor encodings: same z_mean_layer applied to each neighbor slot
  # neigh_z[[k]] shape: (batch, latent_dim)  — encoder mean for k-th neighbor
  neigh_z <- lapply(neigh_enc, function(h) h %>% z_mean_layer())
 
  # ---- Prior networks --------------------------------------------------------
  # Prior log-variance network: aux -> log(sigma^2)  [drives identifiability]
  prior_v <- input_aux
  for (n_units in aux_hidden_units) {
    prior_v <- prior_v %>% keras3::layer_dense(units = n_units, activation = activation)
  }
  prior_log_var       <- prior_v %>% keras3::layer_dense(units = latent_dim)
  prior_log_var_model <- keras3::keras_model(input_aux, prior_log_var)
 
  # Rho network: aux -> (0, 1)  [scaled by rho_max in the loss]
  # Separate network so rho and sigma have independent parameter sets
  rho_v <- input_aux
  for (n_units in aux_hidden_units) {
    rho_v <- rho_v %>% keras3::layer_dense(units = n_units, activation = activation)
  }
  rho_out   <- rho_v %>% keras3::layer_dense(units = 1L, activation = "sigmoid")
  rho_model <- keras3::keras_model(input_aux, rho_out)
 
  # ---- Optional edge weight model: [u_i, u_j] -> score ----------------------
  # Score is later softmaxed over valid neighbors in the loss function.
  # Taking both u_i and u_j as input ensures asymmetric weights w_ij != w_ji.
  if (learn_edge_weights) {
    edge_scores <- lapply(seq_len(max_n_adj), function(k) {
      # Concatenate aux of target i with aux of k-th neighbor j
      edge_in_k <- keras3::layer_concatenate(list(input_aux, neigh_aux_inputs[[k]]))
      ew_h      <- edge_in_k
      for (n_units in edge_hidden_units) {
        ew_h <- ew_h %>% keras3::layer_dense(units = n_units, activation = activation)
      }
      # Scalar score per (i, j) pair
      ew_h %>% keras3::layer_dense(units = 1L)   # (batch, 1)
    })
    # Concatenate to (batch, max_n_adj)
    edge_scores_concat <- keras3::layer_concatenate(edge_scores)
  } else {
    edge_scores_concat <- NULL
  }
 
  # ---- Sampling and decoder --------------------------------------------------
  sampling_layer <- switch(source_dist,
    "gaussian" = sampling_gaussian(p = latent_dim),
    "laplace"  = sampling_laplace(p = latent_dim)
  )
  z_sample <- z_mean_and_var %>% sampling_layer()
 
  x_decoded   <- z_sample
  input_dec   <- keras3::layer_input(latent_dim)
  dec_h       <- input_dec
  for (n_units in rev(hidden_units)) {
    dec_layer <- keras3::layer_dense(units = n_units, activation = activation)
    x_decoded <- x_decoded %>% dec_layer()
    dec_h     <- dec_h     %>% dec_layer()
  }
  out_layer <- keras3::layer_dense(units = p)
  x_decoded <- x_decoded %>% out_layer()
  dec_h     <- dec_h     %>% out_layer()
  decoder   <- keras3::keras_model(input_dec, dec_h)
 
  # ---- Final output concatenation --------------------------------------------
  # Layout (all columns the loss function will parse):
  #   1 : p                                         x_decoded
  #   p+1 : p+L                                     z_sample          (L = latent_dim)
  #   p+L+1 : p+2L                                  z_mean
  #   p+2L+1 : p+3L                                 z_log_var
  #   p+3L+1 : p+4L                                 prior_log_var
  #   p+4L+1                                        rho_out            (1 column)
  #   p+4L+2 : p+4L+1+M*L                           neigh_z concat     (M = max_n_adj)
  #   p+4L+M*L+2 : p+4L+M*L+1+M                    neigh_valid
  #   (if learn_edge_weights) p+4L+M*L+M+2 : +M     edge_scores
 
  # Concatenate neighbor z encodings: (batch, max_n_adj * latent_dim)
  neigh_z_concat <- keras3::layer_concatenate(neigh_z)
 
  output_parts <- list(
    x_decoded,
    z_sample,
    z_mean,
    z_log_var,
    prior_log_var,
    rho_out,
    neigh_z_concat,
    input_neigh_valid
  )
  if (learn_edge_weights) output_parts <- append(output_parts, list(edge_scores_concat))
  final_output <- keras3::layer_concatenate(output_parts)
 
  # ---- Full VAE model --------------------------------------------------------
  all_inputs <- c(
    list(input_data, input_aux),
    neigh_data_inputs,
    neigh_aux_inputs,
    list(input_neigh_valid)
  )
  vae <- keras3::keras_model(all_inputs, final_output)
 
  # ---- Loss function ---------------------------------------------------------
  # Mirrors iVAEar loss: parse the concatenated output, reconstruct the prior
  # mean from neighbor encodings, compute ELBO.
  L   <- latent_dim   # shorthand captured in closure
  M   <- max_n_adj
  rho <- rho_max
 
  vae_loss <- function(x, res) {
    tf <- tensorflow::tf
 
    # ---- Parse output slices (matching layout above) ----
    x_recon    <- res[, 1:p]
    z_samp     <- res[, (p + 1L):(p + L)]
    z_mn       <- res[, (p + L + 1L):(p + 2L * L)]
    z_lv       <- res[, (p + 2L * L + 1L):(p + 3L * L)]
    prior_lv   <- res[, (p + 3L * L + 1L):(p + 4L * L)]
    rho_v      <- res[, (p + 4L * L + 1L):(p + 4L * L + 1L)]  # (batch, 1)
 
    nz_start   <- p + 4L * L + 2L
    nz_end     <- p + 4L * L + 1L + M * L
    neigh_z_v  <- res[, nz_start:nz_end]           # (batch, M*L)
 
    val_start  <- nz_end + 1L
    val_end    <- nz_end + M
    valid_v    <- res[, val_start:val_end]          # (batch, M)  binary
 
    # ---- Compute edge weights -----------------------------------------------
    if (learn_edge_weights) {
      es_start <- val_end + 1L
      es_end   <- val_end + M
      esc_v    <- res[, es_start:es_end]            # (batch, M) raw scores
 
      # Mask out invalid neighbors before softmax
      big_neg      <- (1.0 - tf$cast(valid_v, "float32")) * 1e9
      attn_weights <- tf$nn$softmax(esc_v - big_neg, axis = 1L)  # (batch, M)
    }
 
    # ---- Weighted sum of neighbor encodings ----------------------------------
    z_neigh_sum <- tf$zeros_like(z_mn)  # (batch, L)
 
    for (k in seq_len(M)) {
      # k-th neighbor encoding: (batch, L)
      z_k <- neigh_z_v[, ((k - 1L) * L + 1L):(k * L)]
 
      if (learn_edge_weights) {
        # Learned attention weight for k-th neighbor: (batch, 1)
        w_k <- tf$expand_dims(attn_weights[, k], axis = 1L)
      } else {
        # Uniform: weight = validity flag, will be divided by count below
        w_k <- tf$expand_dims(
          tf$cast(valid_v[, k], "float32"),
          axis = 1L
        )
      }
      z_neigh_sum <- z_neigh_sum + w_k * z_k
    }
 
    if (!learn_edge_weights) {
      # Divide by number of valid neighbors to get mean
      n_valid      <- tf$maximum(
        tf$reduce_sum(tf$cast(valid_v, "float32"), axis = 1L, keepdims = TRUE),
        1.0
      )
      z_neigh_sum <- z_neigh_sum / n_valid
    }
    # (With softmax weights they already sum to 1 over valid neighbors)
 
    # ---- SAR prior mean: rho(u_i) * neighbor aggregate ----------------------
    prior_mean <- rho_v * rho * z_neigh_sum   # rho_v in (0,1), rho = rho_max scalar
 
    # ---- ELBO terms ----------------------------------------------------------
    log_px_z  <- error_log_pdf(x, x_recon, tensorflow::tf$constant(error_dist_sigma, "float32"))
    log_pz_u  <- source_log_pdf(z_samp, prior_mean, tf$math$exp(prior_lv))
    log_qz_xu <- source_log_pdf(z_samp, z_mn,       tf$math$exp(z_lv))
 
    -tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L)
  }
 
  # ---- Reconstruction accuracy metric (mirrors iVAEar) ----------------------
  metric_reconst_accuracy <- keras3::custom_metric(
    "metric_reconst_accuracy",
    function(x, res) {
      x_recon  <- res[, 1:p]
      log_px_z <- error_log_pdf(
        x, x_recon,
        tensorflow::tf$constant(error_dist_sigma, "float32")
      )
      tensorflow::tf$reduce_mean(log_px_z, -1L)
    }
  )
 
  if (is.null(optimizer)) {
    optimizer <- tensorflow::tf$keras$optimizers$Adam(
      learning_rate = tensorflow::tf$keras$optimizers$schedules$PolynomialDecay(
        lr_start, steps, lr_end, 2
      )
    )
  }
 
  vae %>% keras3::compile(
    optimizer = optimizer,
    loss      = vae_loss,
    metrics   = list(metric_reconst_accuracy)
  )
 
  # ---- Assemble training inputs ----------------------------------------------
  # Order must match all_inputs defined above:
  #   [data, aux, neigh_data_1..M, neigh_aux_1..M, neigh_valid]
  fit_inputs <- c(
    list(data_scaled, aux_data),
    neigh_data_list,
    neigh_aux_list,
    list(neigh_valid_mat)
  )
 
  hist <- vae %>% keras3::fit(
    fit_inputs,
    data_scaled,
    shuffle          = TRUE,
    batch_size       = batch_size,
    epochs           = epochs,
    validation_split = validation_split
  )
 
  # ---- Extract results -------------------------------------------------------
  IC_est    <- keras3::predict(encoder, list(data_scaled, aux_data))
  IC_means  <- colMeans(IC_est)
  IC_sds    <- apply(IC_est, 2, sd)
  IC_scaled <- scale(IC_est, center = IC_means, scale = IC_sds)
 
  result <- list(
    IC_unscaled         = IC_est,
    IC                  = IC_scaled,
    IC_means            = IC_means,
    IC_sds              = IC_sds,
    data_dim            = p,
    sample_size         = n,
    aux_dim             = K,
    encoder             = encoder,
    decoder             = decoder,
    rho_model           = rho_model,
    prior_log_var_model = prior_log_var_model,
    learn_edge_weights  = learn_edge_weights,
    rho_max             = rho_max,
    max_n_adj           = max_n_adj,
    data_means          = data_means,
    data_sds            = data_sds,
    adj_list            = adj_list,
    metrics             = hist,
    call_params         = call_params,
    call                = deparse(sys.call()),
    DNAME               = paste(deparse(substitute(data)))
  )
  class(result) <- "GiVAE"
  return(result)
}
 
 
# ------------------------------------------------------------------------------
# predict.GiVAE
# Mirrors predict.iVAE exactly; requires aux_data for the encode direction
# ------------------------------------------------------------------------------
 
#' @export
#' @method predict GiVAE
predict.GiVAE <- function(object, newdata, aux_data = NULL,
                           IC_to_data = FALSE, ...) {
  if (is.numeric(newdata) && is.null(dim(newdata))) newdata <- t(as.matrix(newdata))
  newdata <- as.matrix(newdata)
 
  if (IC_to_data) {
    # Latent -> observation space: undo IC scaling, decode, undo data scaling
    z_unsc <- sweep(newdata, 2, object$IC_sds,   "*")
    z_unsc <- sweep(z_unsc,  2, object$IC_means, "+")
    x_est  <- as.matrix(object$decoder(z_unsc))
    x_est  <- sweep(x_est, 2, object$data_sds,   "*")
    x_est  <- sweep(x_est, 2, object$data_means, "+")
    x_est  <- as.data.frame(x_est)
    names(x_est) <- paste0("X", seq_len(object$data_dim))
    return(x_est)
 
  } else {
    if (is.null(aux_data)) stop("aux_data must be provided when IC_to_data = FALSE")
    if (is.numeric(aux_data) && is.null(dim(aux_data))) aux_data <- t(as.matrix(aux_data))
    aux_data <- as.matrix(aux_data)
 
    x_cent   <- sweep(newdata,  2, object$data_means, "-")
    x_scaled <- sweep(x_cent,  2, object$data_sds,   "/")
    z_est    <- as.matrix(object$encoder(list(x_scaled, aux_data)))
    z_cent   <- sweep(z_est, 2, object$IC_means, "-")
    z_scaled <- sweep(z_cent, 2, object$IC_sds,   "/")
    z_scaled <- as.data.frame(z_scaled)
    names(z_scaled) <- paste0("IC", seq_len(object$call_params$latent_dim))
    return(z_scaled)
  }
}
 
 
# ------------------------------------------------------------------------------
# print.GiVAE
# ------------------------------------------------------------------------------
 
#' @export
#' @method print GiVAE
print.GiVAE <- function(x, ...) {
  cat("Graph iVAE (GiVAE) — Spatial SAR prior\n")
  cat("  Data:               ", x$DNAME, "\n")
  cat("  Observed dim (P):   ", x$data_dim, "\n")
  cat("  Latent dim (L):     ", x$call_params$latent_dim, "\n")
  cat("  Aux dim (K):        ", x$aux_dim, "\n")
  cat("  Sample size (N):    ", x$sample_size, "\n")
  cat("  Max neighbors (M):  ", x$max_n_adj, "\n")
  cat("  Source dist:        ", x$call_params$source_dist, "\n")
  cat("  Error dist:         ", x$call_params$error_dist, "\n")
  cat("  rho_max:            ", x$call_params$rho_max, "\n")
  cat("  Learn edge weights: ", x$call_params$learn_edge_weights, "\n")
  invisible(x)
}
 
 
# ------------------------------------------------------------------------------
# print.GiVAElaplacian  (adds eigenvector info)
# ------------------------------------------------------------------------------
 
#' @export
#' @method print GiVAElaplacian
print.GiVAElaplacian <- function(x, ...) {
  NextMethod()
  cat("  Laplacian eigvecs (K):", x$K, "\n")
  cat("  Eigenvalues (lambda): ",
      paste(round(x$lambda, 4), collapse = ", "), "\n")
  invisible(x)
}
 