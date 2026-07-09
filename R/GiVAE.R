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
#' @param adj_list   Either a single adjacency list (list of length N, each
#'   element a 1-based integer vector of neighbor indices for that observation)
#'   or a list of such adjacency lists, one per graph.
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
  if (learn_edge_weights) {
    warning("learn_edge_weights is currently ignored in GiVAE; graph aggregation uses graph-specific rho and beta weights.")
  }
  if (is.null(adj_list) || length(adj_list) == 0L) stop("adj_list must not be empty")
  if (is.list(adj_list[[1]]) && length(adj_list[[1]]) > 0L) {
    graph_adj_lists <- adj_list
  } else {
    graph_adj_lists <- list(adj_list)
  }
  if (!all(vapply(graph_adj_lists, is.list, logical(1)))) {
    stop("adj_list must be either a single adjacency list or a list of adjacency lists")
  }
  if (!all(vapply(graph_adj_lists, function(g) length(g) == n, logical(1)))) {
    stop("each graph adjacency list must have length nrow(data)")
  }
  n_graphs <- length(graph_adj_lists)
 
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
  # Build one set of padded neighbor matrices per graph and use a common
  # maximum neighbor slot count across graphs so the model can aggregate them.
  graph_neigh_objs <- lapply(graph_adj_lists, function(g) {
    build_neighbor_matrices(data_scaled, aux_data, g)
  })
  max_n_adj <- max(vapply(graph_neigh_objs, function(obj) obj$max_n_adj, integer(1)))
 
  graph_neigh_data_list <- vector("list", n_graphs)
  graph_neigh_aux_list  <- vector("list", n_graphs)
  graph_neigh_valid_mats <- vector("list", n_graphs)
  for (g in seq_len(n_graphs)) {
    obj <- graph_neigh_objs[[g]]
    if (obj$max_n_adj < max_n_adj) {
      pad_data_list <- obj$neigh_data_list
      pad_aux_list  <- obj$neigh_aux_list
      pad_valid_mat <- obj$neigh_valid_mat
      if (ncol(pad_valid_mat) < max_n_adj) {
        pad_valid_mat <- cbind(
          pad_valid_mat,
          matrix(0L, nrow = n, ncol = max_n_adj - ncol(pad_valid_mat))
        )
      }
      for (k in seq.int(obj$max_n_adj + 1L, max_n_adj)) {
        pad_data_list[[k]] <- matrix(0.0, nrow = n, ncol = p)
        pad_aux_list[[k]]  <- matrix(0.0, nrow = n, ncol = K)
      }
      graph_neigh_data_list[[g]] <- pad_data_list
      graph_neigh_aux_list[[g]]  <- pad_aux_list
      graph_neigh_valid_mats[[g]] <- pad_valid_mat
    } else {
      graph_neigh_data_list[[g]] <- obj$neigh_data_list
      graph_neigh_aux_list[[g]]  <- obj$neigh_aux_list
      graph_neigh_valid_mats[[g]] <- obj$neigh_valid_mat
    }
  }
 
  neigh_valid_mat <- do.call(cbind, graph_neigh_valid_mats)
  if (!is.matrix(neigh_valid_mat)) {
    neigh_valid_mat <- matrix(neigh_valid_mat, nrow = n)
  }
 
  if (!is.null(seed)) tensorflow::tf$keras$utils$set_random_seed(as.integer(seed))
  latent_dim <- as.integer(latent_dim)
 
  # ---- Model input layers ----------------------------------------------------
  input_data  <- keras3::layer_input(p)           # current observation
  input_aux   <- keras3::layer_input(K)            # current auxiliary
 
  # One pair of inputs per graph and neighbor slot
  neigh_data_inputs <- vector("list", n_graphs)
  neigh_aux_inputs  <- vector("list", n_graphs)
  for (g in seq_len(n_graphs)) {
    neigh_data_inputs[[g]] <- lapply(seq_len(max_n_adj), function(k) keras3::layer_input(p))
    neigh_aux_inputs[[g]]  <- lapply(seq_len(max_n_adj), function(k) keras3::layer_input(K))
  }
 
  # Binary validity flags: which graph/slot pairs are filled for each observation
  input_neigh_valid <- keras3::layer_input(n_graphs * max_n_adj)
 
  # ---- Encoder with shared layers applied to current + all neighbor slots ----
  enc_current <- keras3::layer_concatenate(list(input_data, input_aux))
  neigh_enc   <- vector("list", n_graphs)
  for (g in seq_len(n_graphs)) {
    neigh_enc[[g]] <- lapply(seq_len(max_n_adj), function(k) {
      keras3::layer_concatenate(list(neigh_data_inputs[[g]][[k]], neigh_aux_inputs[[g]][[k]]))
    })
  }
 
  for (n_units in hidden_units) {
    shared_enc_layer <- keras3::layer_dense(units = n_units, activation = activation)
    enc_current      <- enc_current %>% shared_enc_layer()
    for (g in seq_len(n_graphs)) {
      neigh_enc[[g]] <- lapply(neigh_enc[[g]], function(h) h %>% shared_enc_layer())
    }
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
  neigh_z <- vector("list", n_graphs)
  for (g in seq_len(n_graphs)) {
    neigh_z[[g]] <- lapply(neigh_enc[[g]], function(h) h %>% z_mean_layer())
  }
 
  # ---- Prior networks --------------------------------------------------------
  # Prior log-variance network: aux -> log(sigma^2)  [drives identifiability]
  prior_v <- input_aux
  for (n_units in aux_hidden_units) {
    prior_v <- prior_v %>% keras3::layer_dense(units = n_units, activation = activation)
  }
  prior_log_var       <- prior_v %>% keras3::layer_dense(units = latent_dim)
  prior_log_var_model <- keras3::keras_model(input_aux, prior_log_var)
 
  # Graph-specific rho and beta heads from aux data.
  # Use a shared base network (shared parameters) and independent final
  # output layers per graph. This keeps most parameters shared while
  # allowing lightweight graph-specific heads.
  rho_outs <- vector("list", n_graphs)
  beta_outs <- vector("list", n_graphs)
  rho_models <- vector("list", n_graphs)
  beta_models <- vector("list", n_graphs)

  # Shared base applied to aux_data
  base_v <- input_aux
  for (n_units in aux_hidden_units) {
    base_v <- base_v %>% keras3::layer_dense(units = n_units, activation = activation)
  }

  for (g in seq_len(n_graphs)) {
    # Independent final heads attached to the shared base
    rho_out_g  <- base_v %>% keras3::layer_dense(units = 1L, activation = "sigmoid", name = paste0("rho_out_g", g))
    beta_out_g <- base_v %>% keras3::layer_dense(units = 1L, name = paste0("beta_out_g", g))

    rho_outs[[g]]  <- rho_out_g
    beta_outs[[g]] <- beta_out_g

    # expose per-graph keras models (they will share the base layers)
    rho_models[[g]]  <- keras3::keras_model(input_aux, rho_out_g)
    beta_models[[g]] <- keras3::keras_model(input_aux, beta_out_g)
  }

  rho_out  <- keras3::layer_concatenate(rho_outs)
  beta_out <- keras3::layer_concatenate(beta_outs)
 
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
  #   p+4L+1 : p+4L+G                               rho_out           (G columns)
  #   p+4L+G+1 : p+4L+2G                            beta_out          (G columns)
  #   p+4L+2G+1 : p+4L+2G+G*M*L                     neigh_z concat    (G graphs)
  #   remaining columns                             neigh_valid        (G*M columns)
  graph_neigh_z_concat <- lapply(seq_len(n_graphs), function(g) {
    keras3::layer_concatenate(neigh_z[[g]])
  })
  neigh_z_concat <- keras3::layer_concatenate(graph_neigh_z_concat)
 
  output_parts <- list(
    x_decoded,
    z_sample,
    z_mean,
    z_log_var,
    prior_log_var,
    rho_out,
    beta_out,
    neigh_z_concat,
    input_neigh_valid
  )
  final_output <- keras3::layer_concatenate(output_parts)
 
  # ---- Full VAE model --------------------------------------------------------
  all_inputs <- c(
    list(input_data, input_aux),
    unlist(neigh_data_inputs, recursive = FALSE),
    unlist(neigh_aux_inputs, recursive = FALSE),
    list(input_neigh_valid)
  )
  vae <- keras3::keras_model(all_inputs, final_output)
 
  # ---- Loss function ---------------------------------------------------------
  # Parse the concatenated output, reconstruct the prior mean from each graph's
  # neighbor encodings, combine them with graph-specific rho and beta weights,
  # and compute the ELBO.
  L   <- latent_dim   # shorthand captured in closure
  G   <- n_graphs
  M   <- max_n_adj
  rho <- rho_max
 
  vae_loss <- function(x, res) {
    tf <- tensorflow::tf
 
    # ---- Parse output slices (matching layout above) ----
    x_recon    <- res[, 1:p]
    z_samp     <- res[, (p + 1L):(p + L)]
    z_mn       <- res[, (p + L + 1L):(p + 2L)]
    z_lv       <- res[, (p + 2L + 1L):(p + 3L)]
    prior_lv   <- res[, (p + 3L + 1L):(p + 4L)]
    rho_v      <- res[, (p + 4L + 1L):(p + 4L + G)]   # (batch, G)
    beta_v     <- res[, (p + 4L + G + 1L):(p + 4L + 2 * G)]  # (batch, G)
 
    neigh_start <- p + 4L + 2 * G + 1L
    neigh_end   <- neigh_start + G * M * L - 1L
    neigh_z_v   <- res[, neigh_start:neigh_end]                # (batch, G*M*L)
 
    valid_start <- neigh_end + 1L
    valid_end   <- valid_start + G * M - 1L
    valid_v     <- res[, valid_start:valid_end]                # (batch, G*M)  binary
 
    beta_weights <- tf$nn$softmax(beta_v, axis = 1L)           # (batch, G)
 
    # ---- Weighted sum of graph-specific neighbor encodings ------------------
    z_neigh_sum_total <- tf$zeros_like(z_mn)  # (batch, L)
 
    for (g in seq_len(G)) {
      z_neigh_sum_g <- tf$zeros_like(z_mn)  # (batch, L)
      for (k in seq_len(M)) {
        slot_idx <- (g - 1L) * M + k
        slot_start <- (slot_idx - 1L) * L + 1L
        slot_end   <- slot_start + L - 1L
        z_k <- neigh_z_v[, slot_start:slot_end]
 
        valid_k <- tf$expand_dims(
          tf$cast(valid_v[, slot_idx], "float32"),
          axis = 1L
        )
        z_neigh_sum_g <- z_neigh_sum_g + valid_k * z_k
      }
 
      n_valid_g <- tf$maximum(
        tf$reduce_sum(tf$cast(valid_v[, ((g - 1L) * M + 1L):(g * M)], "float32"), axis = 1L, keepdims = TRUE),
        1.0
      )
      z_neigh_sum_g <- z_neigh_sum_g / n_valid_g
 
      graph_weight <- tf$expand_dims(beta_weights[, g], axis = 1L)
      rho_g        <- tf$expand_dims(rho_v[, g], axis = 1L)
      z_neigh_sum_total <- z_neigh_sum_total + graph_weight * rho_g * z_neigh_sum_g
    }
 
    # ---- SAR prior mean: rho_max * sum_g [rho_g * beta_g * z_neigh_sum_g] ----
    prior_mean <- rho * z_neigh_sum_total
 
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
  #   [data, aux, neigh_data_graph1_slot1..graphG_slotM,
  #    neigh_aux_graph1_slot1..graphG_slotM, neigh_valid]
  neigh_data_fit_list <- unlist(graph_neigh_data_list, recursive = FALSE)
  neigh_aux_fit_list  <- unlist(graph_neigh_aux_list, recursive = FALSE)
  fit_inputs <- c(
    list(data_scaled, aux_data),
    neigh_data_fit_list,
    neigh_aux_fit_list,
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
  IC_est    <- predict(encoder, list(data_scaled, aux_data))
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
    rho_models          = rho_models,
    beta_models         = beta_models,
    prior_log_var_model = prior_log_var_model,
    learn_edge_weights  = FALSE,
    rho_max             = rho_max,
    max_n_adj           = max_n_adj,
    n_graphs            = n_graphs,
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
  cat("  Graphs (G):         ", x$n_graphs, "\n")
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
 