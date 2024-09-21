anisotropic_distance_batch <- function(h, A) {
  # h is a (M x 10 x 2) tensor, A is a (2 x 2) matrix
  h_A <- tf$matmul(h, tf$linalg$inv(A)) # Apply inverse of A to all vectors
  h_A <- tf$reduce_sum(h * h_A, axis = -1) # Compute h^T A^-1 h
  return(tf$sqrt(h_A)) # Output shape is (M x 10)
}

# Matern covariance function (vectorized)
matern_covariance_batch <- function(h_A, sigma, rho, nu) {
  scaled_h <- sqrt(2 * nu) * h_A / rho
  matern_cov <- sigma^2 * ((2^(1-nu)) / tf$math$lgamma(nu)) * (scaled_h^nu) * tfp$math$bessel_kve(v = nu, z = scaled_h)
  matern_cov <- tf$where(tf$equal(scaled_h, 0), sigma^2, matern_cov)
  return(matern_cov) # Shape (M x 10)
}

KrigingLayer(keras$layers$Layer) %py_class% {
    initialize <- function(S, coords, X, aux_dim, max_n_adj, adj_tensor, adj_coords, aux_tensor, encoder) {
        super$initialize()
        self$S <- S
        self$M <- aux_dim
        self$X <- X
        self$coords <- coords
        self$max_n_adj <- max_n_adj
        self$adj_tensor <- adj_tensor
        self$aux_tensor <- aux_tensor
        self$adj_coords <- adj_coords
        self$edge_attr_tensor <- edge_attr_tensor
        self$encoder <- encoder
        self$edge_model <- edge_model
    }

    call <- function(batch_indices) {
        batch_size <- tf$shape(batch_indices)[1L]
        batch_indices <- tf$reshape(batch_indices, list(batch_size))
        batch_indices <- tf$cast(batch_indices, "int32")
        adj_i <- tf$gather(self$adj_tensor, batch_indices, axis = 0L)
        adj_coords_i <- tf$gather(self$adj_coords, batch_indices, axis = 0L)
        aux_i <- tf$gather_nd(self$aux_tensor, batch_indices)

        adj_flat <- tf$reshape(adj_i, shape = list(-1L))
        mask_1d <- tf$expand_dims(tf$not_equal(adj_flat, -1L), c(1L))
        mask <- tf$tile(mask_1d, c(1L, self$S)) # Create a mask to identify valid indices
        adj_temp <- tf$where(tf$not_equal(adj_flat, -1L), adj_flat, 0L)

        X_neigh_i <- tf$gather(self$X, adj_temp)
        aux_neigh_i <- tf$gather(self$aux_tensor, adj_temp)
        X_neigh_i <- tf$where(mask, X_neigh_i, 0L)

        Z_neigh_i <- self$encoder(list(X_neigh_i, aux_neigh_i))

        aux_data_expanded <- tf$expand_dims(aux_i, 2L) # Expands to batch_size x M x 1
        aux_data_tiled <- tf$tile(aux_data_expanded, c(1L, 1L, self$max_n_adj))

        edge_model_input <- tf$concat(list(edge_i, aux_data_tiled), axis = 1L) # Concatenate along the feature dimension
        edge_model_input <- tf$transpose(edge_model_input, perm = c(0L, 2L, 1L))

        edge_model_input_reshaped <- tf$reshape(edge_model_input, c(batch_size * self$max_n_adj, self$L + self$M))

        transformed_edges <- self$edge_model(edge_model_input_reshaped)

        masked_transformed_edges <- tf$where(mask_1d, transformed_edges, -1e9)
        masked_transformed_edges_2d <- tf$reshape(masked_transformed_edges, c(batch_size, self$max_n_adj))
        attention_weights <- tf$nn$softmax(masked_transformed_edges_2d, axis = 1L)
        attention_weights_1d <- tf$reshape(attention_weights, c(batch_size * self$max_n_adj, 1L))
        # Apply attention weights
        weighted_features <- Z_neigh_i * attention_weights_1d # Element-wise multiplication, broadcasted
        weighted_features_reshaped <- tf$reshape(weighted_features, shape = c(batch_size, self$max_n_adj, -1L))
        attended_features <- tf$reduce_sum(weighted_features_reshaped, axis = 1L)

        return(attended_features)
    }
}
attention_layer <- create_layer_wrapper(AttentionLayer)

#library(abind)
#library(deldir)
# Function to pad adjacency indices and edge attributes
pad_adjacency <- function(adj, max_n_adj) {
    pad_length <- max_n_adj - length(adj)
    c(adj, rep(-1, pad_length)) # Pad with -1 for adjacency
}

pad_edge_attributes <- function(attrs, max_n_adj, attr_length) {
    if (length(attrs) < max_n_adj) {
        # Calculate how many to pad
        pad_count <- max_n_adj - length(attrs)
        # Pad each attribute list with zero-vectors of length attr_length
        attrs <- c(attrs, replicate(pad_count, rep(0, attr_length), simplify = FALSE))
    }
    do.call(rbind, attrs) # Combine all attribute vectors into a matrix
}

iVAE_krig_ar <- function(data, aux_data, locations, time_points, latent_dim, neighborhood_indices, prev_indices,
                  hidden_units = c(128, 128, 128), aux_hidden_units = c(128, 128, 128),
                  activation = "leaky_relu", source_dist = "gaussian", validation_split = 0, error_dist = "gaussian",
                  error_dist_sigma = 0.02, optimizer = NULL, lr_start = 0.001, lr_end = 0.0001,
                  steps = 10000, seed = NULL, get_prior_means = TRUE, true_data = NULL, epochs, batch_size) {
    source_dist <- match.arg(source_dist, c("gaussian", "laplace"))
    source_log_pdf <- switch(source_dist,
        "gaussian" = norm_log_pdf,
        "laplace" = laplace_log_pdf
    )
    error_dist <- match.arg(error_dist, c("gaussian", "laplace"))
    error_log_pdf <- switch(error_dist,
        "gaussian" = norm_log_pdf,
        "laplace" = laplace_log_pdf
    )
    call_params <- list(
        latent_dim = latent_dim, source_dist = source_dist, error_dist = error_dist,
        error_dist_sigma = error_dist_sigma, hidden_units = hidden_units,
        aux_hidden_units = aux_hidden_units, activation = "leaky_relu",
        epochs = epochs, batch_size = batch_size, lr_start = lr_start,
        lr_end = lr_end, seed = seed, optimizer = optimizer
    )

    data_means <- colMeans(data)
    data_sds <- apply(data, 2, sd)
    data_cent <- sweep(data, 2, data_means, "-")
    data_scaled <- sweep(data_cent, 2, data_sds, "/")

    if (!is.null(seed)) {
        tf$keras$utils$set_random_seed(as.integer(seed))
    }
    n <- as.integer(dim(data)[1])
    p <- as.integer(dim(data)[2])
    neighborhood_size <- dim(neighborhood_indices)[2]
    dim_aux <- as.integer(dim(aux_data)[2])
    if (n != dim(aux_data)[1]) {
        stop("Observed data and auxiliary data must have same sample size")
    }

    index_matrix <- matrix(0:(n - 1), nrow = n, ncol = 1)

    batch_neighborhood_indices <- tf$gather((neighborhood_indices), batch_indices)
    batch_prev_indices <- tf$gather((prev_indices), batch_indices)
    
    spatial_neighbors_tensor <- tf$gather(data, batch_neighborhood_indices)
    prev_obs_tensor <- tf$gather(data, batch_prev_indices)
    batch_locations <- tf$gather(locations, batch_indices)
    neighbor_locations <- tf$gather(locations, batch_neighborhood_indices)
    
    batch_locations <- tf$reshape(batch_locations, shape = c(length(batch_indices), 1L, 2L))
    squared_diff <- tf$square(batch_locations - neighbor_locations)
    
    squared_distances <- tf$reduce_sum(squared_diff, axis = -1L)
    
    spatial_distances <- tf$sqrt(squared_distances)
    spatial_neighbors_tensor <- tf$gather(data, batch_neighborhood_indices)
    prev_obs_tensor <- tf$gather(data, batch_prev_indices)

    reshaped_neighbors_tensor <- tf$reshape(spatial_neighbors_tensor, shape = c(-1L, p))

    input_aux <- layer_input(dim_aux)
    prior_v <- input_aux
    for (n_units in aux_hidden_units) {
        prior_v <- prior_v %>%
            layer_dense(units = n_units, activation = activation)
    }
    prior_log_var <- prior_v %>% layer_dense(latent_dim)
    # Output parameters for anisotropic Matern covariance
    prior_log_range <- prior_v %>% layer_dense(units = 1, activation = "linear")  # Log of range
    prior_log_sigma <- prior_v %>% layer_dense(units = 1, activation = "linear")  # Log of sigma (variance)
    prior_log_nu <- prior_v %>% layer_dense(units = 1, activation = "linear")  # Log of smoothness parameter nu

    # For anisotropy, we output the 4 elements of a 2x2 matrix A
    prior_A_elements <- prior_v %>% layer_dense(units = 4, activation = "linear")  # 4 elements of matrix A

    # Convert these outputs to their corresponding covariance model parameters
    prior_range <- tf$exp(prior_log_range)
    prior_sigma <- tf$exp(prior_log_sigma)
    prior_nu <- tf$exp(prior_log_nu)
    prior_A <- tf$reshape(prior_A_elements, shape = c(2, 2))  # Reshape to 2x2 matrix

    h <- batch_locations - neighbor_locations  # (batch_size, neighborhood_size, 2)
    # Compute anisotropic distances for each dimension P
    matern_cov_batch_p <- list()
    for (i in 1:p) {
        h_A_p <- anisotropic_distance_batch(h, prior_A[, i])  # Shape (batch_size, neighborhood_size)
        matern_cov_batch_p[[i]] <- matern_covariance_batch(h_A_p, prior_sigma[, i], prior_range[, i], prior_nu[, i])  # Shape (batch_size, neighborhood_size)
    }

    # Stack into (batch_size, neighborhood_size, P)
    matern_cov_batch <- tf$stack(matern_cov_batch_p, axis = -1L)
    
    # Compute neighbor-neighbor differences (neighbor-to-neighbor distances)
    neighbor_diffs_expanded <- neighbor_locations - tf$expand_dims(neighbor_locations, axis = 2L)  # Shape (batch_size, neighborhood_size, neighborhood_size, 2)

    # Compute anisotropic distances and Matern covariances for each dimension P
    neighbor_cov_matrix_p <- list()

    for (i in 1:P) {
        h_A_neighbor_p <- anisotropic_distance_batch(neighbor_diffs_expanded, prior_A[, i])  # Shape (batch_size, neighborhood_size, neighborhood_size)
        neighbor_cov_matrix_p[[i]] <- matern_covariance_batch(h_A_neighbor_p, prior_sigma[, i], prior_range[, i], prior_nu[, i])  # Shape (batch_size, neighborhood_size, neighborhood_size)
    }

    # Stack into (batch_size, neighborhood_size, neighborhood_size, P)
    neighbor_cov_matrix <- tf$stack(neighbor_cov_matrix_p, axis = -1L)

    batch_indices <- c(0L, 1L, 2L, 3L, 4L, 5L)
    input_data <- layer_input(p)
    input <- layer_concatenate(list(input_data, input_aux))
    submodel <- input
    for (n_units in hidden_units) {
        submodel <- submodel %>%
            layer_dense(units = n_units, activation = activation)
    }
    z_mean <- submodel %>% layer_dense(latent_dim)
    z_log_var <- submodel %>% layer_dense(latent_dim)
    z_mean_and_var <- layer_concatenate(list(z_mean, z_log_var))
    encoder <- keras_model(list(input_data, input_aux), z_mean)
    z_log_var_model <- keras_model(list(input_data, input_aux), z_log_var)

    # Feed the reshaped neighbors into the encoder
    encoded_neighbors <- encoder(list(reshaped_neighbors_tensor, input_aux))
    encoded_neighbors <- tf$reshape(encoded_neighbors, shape = c(length(batch_indices), neighborhood_size, latent_dim))

    kriging_prediction_p <- list()

    for (i in 1:P) {
        # Extract covariances for dimension i
        matern_cov_batch_i <- matern_cov_batch[, , i]  # Shape (batch_size, neighborhood_size)
        neighbor_cov_matrix_i <- neighbor_cov_matrix[, , , i]  # Shape (batch_size, neighborhood_size, neighborhood_size)
        
        # Invert the neighbor covariance matrix for dimension i
        neighbor_cov_matrix_inv_i <- tf$linalg$inv(neighbor_cov_matrix_i)  # Shape (batch_size, neighborhood_size, neighborhood_size)

        # Kriging weights: k^T * K^-1
        kriging_weights_i <- tf$matmul(tf$expand_dims(matern_cov_batch_i, axis = 1L), neighbor_cov_matrix_inv_i)  # Shape (batch_size, 1, neighborhood_size)

        # Kriging prediction: k^T * K^-1 * y
        kriging_prediction_i <- tf$matmul(kriging_weights_i, tf$expand_dims(encoded_neighbors[, , i], axis = -1L))  # Shape (batch_size, 1)
        
        kriging_prediction_p[[i]] <- kriging_prediction_i
    }

    # Stack predictions into (batch_size, P)
    kriging_prediction <- tf$concat(kriging_prediction_p, axis = -1L)
    

    
    return(list(
        spatial_neighbors = spatial_neighbors_tensor,
        previous_observations = prev_obs_tensor,
        spatial_distances = spatial_distances,
        batch_neighborhood_indices = batch_neighborhood_indices
    ))
}
