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
        aux_i <- tf$gather(self$aux_tensor, batch_indices)

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

    # Placeholder for batch of indices
    batch_indices <- c(1L, 2L, 3L, 4L, 5L, 6L)
    
    batch_neighborhood_indices <- tf$gather(neighborhood_indices, batch_indices)
    batch_prev_indices <- tf$gather(prev_indices, batch_indices)
    spatial_neighbors_tensor <- tf$gather(data, batch_neighborhood_indices)
    prev_obs_tensor <- tf$gather(data, batch_prev_indices)
    # Gather spatial coordinates for the batch
    batch_coords <- tf$gather(locations, batch_indices)
    neighbor_coords <- tf$gather(locations, tf$reshape(batch_neighborhood_indices, shape = c(-1L)))
    
    # Reshape tensors to ensure correct dimensions for distance calculation
    batch_coords <- tf$reshape(batch_coords, shape = c(length(batch_indices), -1L))
    neighbor_coords <- tf$reshape(neighbor_coords, shape = c(length(batch_indices), -1L, 2L))
    
    # Calculate pairwise distances between each point in the batch and its neighbors (using spatial coordinates)
    batch_coords_expanded <- tf$expand_dims(batch_coords, axis = 1L)
    neighbor_coords_expanded <- tf$expand_dims(neighbor_coords, axis = 2L)
    
    # Compute squared differences for spatial coordinates
    squared_diff <- tf$square(batch_coords_expanded - neighbor_coords_expanded)
    
    # Sum over spatial dimensions (x, y) to get squared Euclidean distance
    squared_distances <- tf$reduce_sum(squared_diff, axis = -1L)
    
    # Compute Euclidean distances for spatial coordinates
    spatial_distances <- tf$sqrt(squared_distances)
    
    # Gather data for observations in the batch and neighbors
    spatial_neighbors_tensor <- tf$gather(data, batch_neighborhood_indices)
    prev_obs_tensor <- tf$gather(data, batch_prev_indices)
    
    return(list(
        spatial_neighbors = spatial_neighbors_tensor,
        previous_observations = prev_obs_tensor,
        spatial_distances = spatial_distances
    ))
}

{
    adj_list <- lapply(edge_list, function(node_list) {
        sapply(node_list, function(edge) edge[[1]])
    })

    edge_attrs_list <- lapply(edge_list, function(node_list) {
        sapply(node_list, function(edge) edge[[2]], simplify = FALSE)
    })

    l <- as.integer(length(edge_attrs_list[[1]][[1]]))
    max_n_adj <- as.integer(max(sapply(adj_list, length)))
    adj_tensor <- tf$transpose(tf$constant(sapply(adj_list, pad_adjacency, max_n_adj = max_n_adj), dtype = tf$int32))
    edge_attr_matrix <- abind(lapply(edge_attrs_list, pad_edge_attributes, max_n_adj = max_n_adj, attr_length = l), along = 3)
    edge_attr_matrix <- aperm(edge_attr_matrix, c(3, 2, 1))
    edge_attr_tensor <- tf$constant(edge_attr_matrix, dtype = tf$float32)
    aux_tensor <- tf$constant(aux_data, dtype = tf$float32)

    data_means <- colMeans(data)
    data_sds <- apply(data, 2, sd)
    data_cent <- sweep(data, 2, data_means, "-")
    data_scaled <- sweep(data_cent, 2, data_sds, "/")
    data_scaled_tensor <- tf$constant(data_scaled, dtype = tf$float32)

    if (!is.null(seed)) {
        tf$keras$utils$set_random_seed(as.integer(seed))
    }
    n <- as.integer(dim(data)[1])
    p <- as.integer(dim(data)[2])
    dim_aux <- as.integer(dim(aux_data)[2])
    if (n != dim(aux_data)[1]) {
        stop("Observed data and auxiliary data must have same sample size")
    }

    index_matrix <- matrix(0:(n - 1), nrow = n, ncol = 1)

    input_aux <- layer_input(dim_aux)
    prior_v <- input_aux
    for (n_units in aux_hidden_units) {
        prior_v <- prior_v %>%
            layer_dense(units = n_units, activation = activation)
    }
    prior_log_var <- prior_v %>% layer_dense(latent_dim)
    prior_log_var_model <- keras_model(input_aux, prior_log_var)

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

    edge_input_size <- l + dim_aux
    edge_model_input <- layer_input(edge_input_size)
    edge_model_output <- edge_model_input
    for (n_units in edge_model_hidden_units) {
        edge_model_output <- edge_model_output %>%
            layer_dense(units = n_units, activation = activation)
    }
    edge_model_output <- edge_model_output %>% layer_dense(1)
    edge_model <- keras_model(edge_model_input, edge_model_output)

    attention <- attention_layer(
        S = p, L = l, X = data_scaled_tensor, aux_dim = dim_aux,
        max_n_adj = max_n_adj, adj_tensor = adj_tensor,
        edge_attr_tensor = edge_attr_tensor, aux_tensor = aux_tensor,
        encoder = encoder, edge_model = edge_model
    )

    input_aux_mean <- layer_input(1L)
    output_aux_mean <- input_aux_mean %>% attention()
    prior_mean_model <- keras_model(input_aux_mean, output_aux_mean)

    sampling_layer <- switch(source_dist,
        "gaussian" = sampling_gaussian(p = latent_dim),
        "laplace" = sampling_laplace(p = latent_dim)
    )
    z <- z_mean_and_var %>% sampling_layer()

    x_decoded_mean <- z
    input_decoder <- layer_input(latent_dim)
    output_decoder <- input_decoder
    for (n_units in rev(hidden_units)) {
        dense_layer <- layer_dense(units = n_units, activation = activation)
        x_decoded_mean <- x_decoded_mean %>%
            dense_layer()
        output_decoder <- output_decoder %>% dense_layer()
    }
    out_layer <- layer_dense(units = p)
    x_decoded_mean <- x_decoded_mean %>% out_layer()
    output_decoder <- output_decoder %>% out_layer()
    decoder <- keras_model(input_decoder, output_decoder)
    final_output <- layer_concatenate(list(x_decoded_mean, z, z_mean_and_var, output_aux_mean, prior_log_var))

    vae <- keras_model(list(input_data, input_aux, input_aux_mean), final_output)
    vae_loss <- function(x, res) {
        x_mean <- res[, 1:p]
        z_sample <- res[, (1 + p):(p + latent_dim)]
        z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
        z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
        prior_mean_v <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
        prior_log_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
        log_px_z <- error_log_pdf(x, x_mean, tf$constant(error_dist_sigma, "float32"))
        log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
        log_pz_u <- source_log_pdf(z_sample, prior_mean_v, tf$math$exp(prior_log_v))

        return(-tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L))
    }
    if (is.null(optimizer)) {
        optimizer <- tf$keras$optimizers$Adam(learning_rate = tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2))
    }

    metric_reconst_accuracy <- custom_metric("metric_reconst_accuracy", function(x, res) {
        x_mean <- res[, 1:p]
        log_px_z <- error_log_pdf(x, x_mean, tf$constant(error_dist_sigma, "float32"))
        return(tf$reduce_mean(log_px_z, -1L))
    })

    metric_kl_vae <- custom_metric("metric_kl_vae", function(x, res) {
        z_sample <- res[, (1 + p):(p + latent_dim)]
        z_mean <- res[, (p + latent_dim + 1):(p + 2 * latent_dim)]
        z_logvar <- res[, (p + 2 * latent_dim + 1):(p + 3 * latent_dim)]
        prior_mean_v <- res[, (p + 3 * latent_dim + 1):(p + 4 * latent_dim)]
        prior_log_v <- res[, (p + 4 * latent_dim + 1):(p + 5 * latent_dim)]
        log_qz_xu <- source_log_pdf(z_sample, z_mean, tf$math$exp(z_logvar))
        log_pz_u <- source_log_pdf(z_sample, prior_mean_v, tf$math$exp(prior_log_v))
        return(-tf$reduce_mean((log_pz_u - log_qz_xu), -1L))
    })

    vae %>% compile(
        optimizer = optimizer,
        loss = vae_loss,
        metrics = list(metric_reconst_accuracy, metric_kl_vae)
    )
    MCCs <- numeric(epochs)
    if (!is.null(true_data)) {
        for (i in 1:epochs) {
            hist <- vae %>% fit(list(data_scaled, aux_data, index_matrix), data_scaled, validation_split = validation_split, shuffle = TRUE, batch_size = batch_size, epochs = 1)
            IC_estimates <- predict(encoder, list(data_scaled, aux_data))
            MCCs[i] <- absolute_mean_correlation(cor(IC_estimates, true_data))
        }
    } else {
        hist <- vae %>% fit(list(data_scaled, aux_data, index_matrix), data_scaled, validation_split = validation_split, shuffle = TRUE, batch_size = batch_size, epochs = epochs)
    }
    IC_estimates <- predict(encoder, list(data_scaled, aux_data))
    obs_estimates <- predict(decoder, IC_estimates)
    IC_log_vars <- predict(z_log_var_model, list(data_scaled, aux_data))
    prior_means <- predict(prior_mean_model, index_matrix)
    prior_log_vars <- predict(prior_log_var_model, aux_data)
    log_px_z <- error_log_pdf(tf$constant(data_scaled, "float32"), tf$cast(obs_estimates, "float32"), tf$constant(error_dist_sigma, "float32"))
    log_qz_xu <- source_log_pdf(tf$cast(IC_estimates, "float32"), tf$cast(IC_estimates, "float32"), tf$math$exp(tf$cast(IC_log_vars, "float32")))
    log_pz_u <- source_log_pdf(tf$cast(IC_estimates, "float32"), tf$cast(prior_means, "float32"), tf$math$exp(tf$cast(prior_log_vars, "float32")))
    elbo <- tf$reduce_mean(log_px_z + log_pz_u - log_qz_xu, -1L)
    IC_means <- colMeans(IC_estimates)
    IC_sds <- apply(IC_estimates, 2, sd)
    IC_estimates_cent <- sweep(IC_estimates, 2, IC_means, "-")
    IC_estimates_scaled <- sweep(IC_estimates_cent, 2, IC_sds, "/")
    IC_vars <- exp(IC_log_vars)
    IC_vars_scaled <- sweep(IC_vars, 2, IC_sds, "/")
    prior_means_scaled <- NULL
    prior_vars_scaled <- NULL
    if (get_prior_means) {
        prior_means_cent <- sweep(prior_means, 2, IC_means, "-")
        prior_means_scaled <- sweep(prior_means_cent, 2, IC_sds, "/")
        prior_vars <- exp(prior_log_vars)
        prior_vars_scaled <- sweep(prior_vars, 2, IC_sds, "/")
    }

    iVAE_object <- list(
        IC_unscaled = IC_estimates, IC = IC_estimates_scaled, IC_vars = IC_vars_scaled, elbo = as.numeric(elbo), reconst_acc = as.numeric(tf$reduce_mean(log_px_z, -1L)),
        prior_means = prior_means_scaled, prior_vars = prior_vars_scaled, data_dim = p,
        sample_size = n, prior_mean_model = prior_mean_model, prior_log_var_model = prior_log_var_model, call_params = call_params,
        aux_dim = dim_aux, encoder = encoder, decoder = decoder, data_means = data_means,
        data_sds = data_sds, IC_means = IC_means, IC_sds = IC_sds, MCCs = MCCs, call = deparse(sys.call()),
        DNAME = paste(deparse(substitute(data))), metrics = hist, orig_data = data_scaled, aux_data = aux_data
    )

    class(iVAE_object) <- "iVAE"
    return(iVAE_object)
}

form_neighborhoods <- function(spatial_locations, time_points, neighborhood_size) {
    N <- length(time_points)
    neighborhood_indices <- matrix(NA, nrow = N, ncol = neighborhood_size)
    unique_time_points <- unique(time_points)
    
    for (cur_time in unique_time_points) {
        time_inds <- which(time_points == cur_time)
        spatial_locs_cur_time <- spatial_locations[time_inds, , drop = FALSE]
        dist_matrix <- rdist::rdist(spatial_locs_cur_time)
        nearest_neighbors <- apply(as.matrix(dist_matrix), 1, function(dists) {
            order(dists)[1:neighborhood_size]
        })        
        neighborhood_indices[time_inds, ] <- time_inds[nearest_neighbors]
    }
    
    return(neighborhood_indices)
}

form_prev_obs_data <- function(spatial_locations, time_points, ar) {
    N <- length(time_points)
    prev_obs_indices <- matrix(NA, nrow = N, ncol = ar)
    unique_spatial_locs <- unique(spatial_locations)
    for (spatial_loc in seq_len(nrow(unique_spatial_locs))) {
        cur_spatial_loc <- unique_spatial_locs[spatial_loc, , drop = FALSE]
        spatial_inds <- which(apply(spatial_locations, 1, function(loc) all(loc == cur_spatial_loc)))   
        spatial_time_points <- time_points[spatial_inds]
        for (i in spatial_inds) {
            cur_time <- time_points[i]
            prev_time_inds <- which(spatial_time_points < cur_time)
            if (length(prev_time_inds) < ar) {
                prev_obs_indices[i, ] <- rep(i, ar)
            } else {
                prev_obs_indices[i, ] <- tail(spatial_inds[prev_time_inds], ar)
            }
        }
    }
    
    return(prev_obs_indices)
}


iVAE_krig_ar_radial <- function(data, spatial_locations, time_points, latent_dim, 
    elevation = NULL, test_inds = NULL, neighborhood_size = 5, ar = 1, spatial_dim = 2, spatial_basis = c(2, 9), 
    temporal_basis = c(9, 17, 37), elevation_basis = NULL, seasonal_period = NULL, 
    spatial_kernel = "gaussian", epochs, batch_size, ...) {
    
    aux_data_obj <- form_radial_aux_data(spatial_locations, time_points, elevation, test_inds, spatial_dim, spatial_basis, temporal_basis, elevation_basis, seasonal_period, spatial_kernel)

    neighborhood_indices <- form_neighborhoods(spatial_locations, time_points, neighborhood_size)

    prev_obs_indices <- form_prev_obs_data(spatial_locations, time_points, ar)
    
    resVAE <- iVAE_krig_ar(data, aux_data_obj$aux_data, spatial_locations, time_points, latent_dim, neighborhood_indices, prev_obs_indices, epochs = epochs, batch_size = batch_size, get_prior_means = FALSE, ...)
    class(resVAE) <- c("iVAE_krig_ar1", class(resVAE))
    resVAE$spatial_basis <- spatial_basis
    resVAE$temporal_basis <- temporal_basis
    resVAE$elevation_basis <- elevation_basis
    resVAE$spatial_kernel <- aux_data_obj$spatial_kernel
    resVAE$min_time_point <- aux_data_obj$min_time_point
    resVAE$max_time_point <- aux_data_obj$max_time_point
    resVAE$min_elevation <- aux_data_obj$min_elevation
    resVAE$max_elevation <- aux_data_obj$max_elevation
    resVAE$spatial_dim <- spatial_dim
    return(resVAE)
}

p <- 3
n_time <- 100
n_spat <- 50
coords_time <- cbind(
    rep(runif(n_spat), n_time),
    rep(runif(n_spat), n_time), rep(1:n_time, each = n_spat)
)
data_obj <- generate_nonstationary_spatio_temporal_data_by_segments(
    n_time,
    n_spat, p, 5, 10, coords_time
)
iVAE_krig_ar_radial(as.matrix(data_obj$data), coords_time[, 1:2], coords_time[, 3], 3, epochs = 1, batch_size = 60)

GiVAE_spatial_seg <- function(
    data, locations, segment_sizes,
    joint_segment_inds = rep(1, length(segment_sizes)), latent_dim,
    test_inds = NULL, epochs, batch_size, ...) {
    n <- nrow(data)
    location_mins <- apply(locations, 2, min)
    locations_zero <- sweep(locations, 2, location_mins, "-")
    location_maxs <- apply(locations_zero, 2, max)
    aux_data <- matrix(nrow = n, ncol = 0)
    for (i in unique(joint_segment_inds)) {
        inds <- which(joint_segment_inds == i)
        labels <- rep(0, n)
        lab <- 1
        loop_dim <- function(j, sample_inds) {
            ind <- inds[j]
            seg_size <- segment_sizes[ind]
            seg_limits <- seq(0, (location_maxs[ind]), seg_size)
            for (coord in seg_limits) {
                cur_inds <- which(locations[sample_inds, ind] >= coord &
                    locations[sample_inds, ind] < coord + seg_size)
                cur_sample_inds <- sample_inds[cur_inds]
                if (j == length(inds)) {
                    labels[cur_sample_inds] <<- lab
                    lab <<- lab + 1
                } else {
                    loop_dim(j + 1, cur_sample_inds)
                }
            }
        }
        loop_dim(1, 1:n)
        labels <- as.numeric(as.factor(labels)) # To ensure that
        # empty segments are reduced
        aux_data <- cbind(aux_data, model.matrix(~ 0 + as.factor(labels)))
    }

    locations_new <- sweep(locations_zero, 2, location_maxs, "/")

    triangulation <- deldir(locations_new[, 1], locations_new[, 2])
    triangles <- triangulation$delsgs

    edge_list <- list()

    edge_dists <- apply(triangles, 1, function(row) sqrt(sum((row[1:2] - row[3:4])^2)))
    edge_dirs_n1n2 <- apply(triangles, 1, function(row) atan2(row[4] - row[2], row[3] - row[1]))
    edge_dirs_n2n1 <- apply(triangles, 1, function(row) atan2(row[2] - row[4], row[1] - row[3]))
    for (i in seq_len(nrow(locations))) {
        node_i_neigh_idxs_n1n2 <- which(triangles[, 5] == i)
        node_i_neigh_idxs_n2n1 <- which(triangles[, 6] == i)

        node_i_list <- list()
        for (idx in seq_along(node_i_neigh_idxs_n1n2)) {
            row_idx <- node_i_neigh_idxs_n1n2[idx]
            node_idx <- triangles[row_idx, 6]
            node_i_list[[idx]] <- list(node_idx - 1, c(edge_dists[row_idx], edge_dirs_n1n2[row_idx]))
        }
        node_i_list_length <- length(node_i_list)
        for (idx in seq_along(node_i_neigh_idxs_n2n1)) {
            row_idx <- node_i_neigh_idxs_n2n1[idx]
            node_idx <- triangles[row_idx, 5]
            node_i_list[[idx + node_i_list_length]] <- list(node_idx - 1, c(edge_dists[row_idx], edge_dirs_n2n1[row_idx]))
        }
        edge_list <- append(edge_list, list(node_i_list))
    }
    resVAE <- GiVAE(data, aux_data, latent_dim, edge_list, epochs = epochs, batch_size = batch_size, get_prior_means = FALSE, ...)
    class(resVAE) <- c("GiVAEspatial", class(resVAE))
    resVAE$min_coords <- location_mins
    resVAE$max_coords <- location_maxs
    # resVAE$phi_maxs <- phi_maxs
    resVAE$spatial_dim <- dim(locations)[2]
    return(resVAE)
}


edge_list_radius <- function(coords, radius) {
    n <- dim(coords)[1]
    d <- as.matrix(distances::distances(coords))
    d_scaled <- d / radius
    edge_list <- list()
    for (i in seq_len(n)) {
        neigh_idxs <- as.numeric(which(d[i, ] != 0 & d[i, ] <= radius))
        node_i_list <- list()
        for (j in seq_along(neigh_idxs)) {
            neigh_i <- neigh_idxs[j]
            node_start <- coords[i, ]
            node_end <- coords[neigh_i, ]
            edge_dir <- atan2(node_end[2] - node_start[2], node_end[1] - node_start[1])
            edge_length <- d_scaled[i, neigh_i]
            node_i_list[[j]] <- list(neigh_i - 1, c(edge_length, edge_dir))
        }
        edge_list <- append(edge_list, list(node_i_list))
    }
    return(edge_list)
}

GiVAE_spatial2 <- function(data, locations, latent_dim, neighborhood_radius, num_basis = c(2, 9), kernel = "gaussian", epochs, batch_size, ...) {
    kernel <- match.arg(kernel, c("gaussian", "wendland"))
    spatial_dim <- dim(locations)[2]
    N <- dim(data)[1]
    min_coords <- apply(locations, 2, min)
    locations_new <- sweep(locations, 2, min_coords, "-")
    max_coords <- apply(locations_new, 2, max)
    locations_new <- sweep(locations_new, 2, max_coords, "/")
    knots_1d <- sapply(num_basis, FUN = function(i) seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i))
    phi_all <- matrix(0, ncol = 0, nrow = N)
    for (i in seq_along(num_basis)) {
        theta <- 1 / num_basis[i] * 2.5
        knot_list <- replicate(spatial_dim, knots_1d[[i]], simplify = FALSE)
        knots <- as.matrix(expand.grid(knot_list))
        phi <- cdist(locations_new, knots) / theta
        dist_leq_1 <- phi[which(phi <= 1)]
        dist_g_1_ind <- which(phi > 1)
        phi[which(phi <= 1)] <- switch(kernel,
            gaussian = gaussian_kernel(dist_leq_1),
            wendland = wendland_kernel(dist_leq_1)
        )
        phi[dist_g_1_ind] <- 0
        phi_all <- cbind(phi_all, phi)
    }
    aux_data <- phi_all

    edge_list <- edge_list_radius(locations, neighborhood_radius)
    resVAE <- GiVAE(data, aux_data, latent_dim, edge_list, epochs = epochs, batch_size = batch_size, get_prior_means = FALSE, ...)
    class(resVAE) <- c("GiVAEspatial", class(resVAE))
    resVAE$min_coords <- min_coords
    resVAE$max_coords <- max_coords
    resVAE$num_basis <- num_basis
    # resVAE$phi_maxs <- phi_maxs
    resVAE$spatial_dim <- dim(locations)[2]
    return(resVAE)
}