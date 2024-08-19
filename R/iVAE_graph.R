AttentionLayer(keras$layers$Layer) %py_class% {
    initialize <- function(S, L, X, aux_dim, max_n_adj, adj_tensor, edge_attr_tensor, aux_tensor, encoder, edge_model) {
        super$initialize()
        self$S <- S
        self$M <- aux_dim
        self$L <- L
        self$X <- X
        self$max_n_adj <- max_n_adj
        self$adj_tensor <- adj_tensor
        self$aux_tensor <- aux_tensor
        self$edge_attr_tensor <- edge_attr_tensor
        self$encoder <- encoder
        self$edge_model <- edge_model
    }

    call <- function(batch_indices) {
        batch_size <- tf$shape(batch_indices)[1L]
        batch_indices <- tf$reshape(batch_indices, list(batch_size))
        batch_indices <- tf$cast(batch_indices, "int32")
        adj_i <- tf$gather(self$adj_tensor, batch_indices, axis = 0L)
        edge_i <- tf$gather(self$edge_attr_tensor, batch_indices, axis = 0L)
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

library(abind)
library(deldir)
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

GiVAE <- function(data, aux_data, latent_dim, edge_list,
                  hidden_units = c(128, 128, 128), aux_hidden_units = c(128, 128, 128),
                  edge_model_hidden_units = c(128, 128, 128),
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
        optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate = tf$keras$optimizers$schedules$PolynomialDecay(lr_start, steps, lr_end, 2))
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

    mse_vae <- custom_metric("mse_vae", function(x, res) {
        x_mean <- res[, 1:p]
        return(metric_mean_squared_error(x, x_mean))
    })

    vae %>% compile(
        optimizer = optimizer,
        loss = vae_loss,
        metrics = list(metric_reconst_accuracy, metric_kl_vae, mse_vae)
    )
    MCCs <- numeric(epochs)
    if (!is.null(true_data)) {
        for (i in 1:epochs) {
            hist <- vae %>% fit(list(data_scaled, aux_data, index_matrix), data_scaled, validation_split = validation_split, shuffle = TRUE, batch_size = batch_size, epochs = 1, seed = seed)
            IC_estimates <- predict(encoder, list(data_scaled, aux_data))
            MCCs[i] <- absolute_mean_correlation(cor(IC_estimates, true_data))
        }
    } else {
        hist <- vae %>% fit(list(data_scaled, aux_data, index_matrix), data_scaled, validation_split = validation_split, shuffle = TRUE, batch_size = batch_size, epochs = epochs, seed = seed)
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

GiVAE_spatial_radial <- function(data, locations, latent_dim, num_basis = c(2, 9), kernel = "gaussian", epochs, batch_size, ...) {
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

    triangulation <- deldir(locations_new[, 1], locations_new[, 2])
    triangles <- triangulation$delsgs

    edge_list <- list()

    edge_dists <- apply(triangles, 1, function(row) sqrt(sum((row[1:2] - row[3:4])^2)))
    edge_dirs_n1n2 <- apply(triangles, 1, function(row) atan2(row[4] - row[2], row[3] - row[1]))
    edge_dirs_n2n1 <- apply(triangles, 1, function(row) atan2(row[2] - row[4], row[1] - row[3]))
    for (i in seq_len(nrow(locations_new))) {
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
    resVAE$min_coords <- min_coords
    resVAE$max_coords <- max_coords
    resVAE$num_basis <- num_basis
    # resVAE$phi_maxs <- phi_maxs
    resVAE$spatial_dim <- dim(locations)[2]
    return(resVAE)
}


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

#' Plotting an Object of Class iVAE
#' @description \code{plot} method for the class \code{iVAE}.
#' @param x Object of class \code{iVAE}
#' @param IC_inds A vector providing indices of ICs which are plotted.
#' @param sample_inds A vector providing sample indices which are plotted.
#' @param unscaled A boolean determining if the unscaled ICs are plotted
#' or not.
#' @param type linetype for plot.default function.
#' @param xlab xlab for the bottom most plot.
#' @param ylabs A vector providing ylab params for the plots.
#' @param colors A vector providing colors for the plots.
#' @param oma A \code{oma} parameter for the function \code{par}.
#' See \code{\link[graphics]{par}}
#' @param mar A \code{mar} parameter for the function \code{par}.
#' See \code{\link[graphics]{par}}
#' @param ... Further parameters for all plot.default methods.
#' @returns
#' No return value.
#' @details
#' Plots the components provided by the parameter \code{IC_inds}. A sample
#' subset can be provided by parameter \code{sample_inds}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'     for (seg in 1:n_segments) {
#'         start_ind <- (seg - 1) * n_per_segment + 1
#'         end_ind <- seg * n_per_segment
#'         latent_data[start_ind:end_ind, i] <- rnorm(
#'             n_per_segment,
#'             runif(1, -5, 5), runif(1, 0.1, 5)
#'         )
#'     }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' plot(res)
#' @export
#' @author Mika Sipilä
#' @method plot iVAE
plot.iVAE <- function(
    x, IC_inds = 1:x$call_params$latent_dim,
    sample_inds = 1:dim(x$IC)[1], unscaled = FALSE, type = "l",
    xlab = "", ylabs = c(), colors = c(), oma = c(1, 1, 0, 0),
    mar = c(2, 2, 1, 1), ...) {
    oldpar <- par(no.readonly = TRUE)
    on.exit(par(oldpar))

    if (unscaled) {
        ICs <- x$IC_unscaled[sample_inds, IC_inds]
    } else {
        ICs <- x$IC[sample_inds, IC_inds]
    }
    p <- dim(ICs)[2]
    par(
        mfrow = c(p, 1),
        oma = oma,
        mar = mar,
        mgp = c(2, 0.5, 0),
        xpd = NA
    )
    for (i in 1:p) {
        if (length(ylabs) < i) ylabs[i] <- paste0("IC ", IC_inds[i])
    }
    for (i in 1:p) {
        if (is.null(colors)) {
            col <- 1
        } else {
            col <- ifelse(is.na(colors[i]), 1, colors[i])
        }
        plot(ICs[, i],
            type = type, xlab = ifelse(i == p, xlab, ""),
            ylab = ylabs[i], col = col, ...
        )
    }
}

#' Printing an Object of Class iVAE
#' @description \code{print} method for the class \code{iVAE}.
#' @param x Object of class \code{iVAE}
#' @param ... Further parameters for \code{print.default}.
#' @returns
#' No return value.
#' @details
#' Prints details about the object of class \code{iVAE}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'     for (seg in 1:n_segments) {
#'         start_ind <- (seg - 1) * n_per_segment + 1
#'         end_ind <- seg * n_per_segment
#'         latent_data[start_ind:end_ind, i] <- rnorm(
#'             n_per_segment,
#'             runif(1, -5, 5), runif(1, 0.1, 5)
#'         )
#'     }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' print(res)
#' @export
#' @author Mika Sipilä
#' @method print iVAE
print.iVAE <- function(x, ...) {
    cat("Call:", x$call)
    cat("\nData name:", x$DNAME)
    cat("\nDimension of observed data:", x$p)
    cat("\nLatent dimension:", x$call_params$latent_dim)
    cat("\nDimension of auxiliary data:", x$aux_dim)
    cat("\nSample size:", x$n)
    cat("\nSource density model: ", x$call_params$source_dist)
    cat("\nError density model: ", x$call_params$error_dist)
    cat("\nError sigma: ", x$call_params$error_dist_sigma)
}

#' Summarize an Object of Class iVAE
#' @description \code{summary} method for the class \code{iVAE}.
#' @param object Object of class \code{iVAE}
#' @param ... Further parameters for \code{summary.default}.
#' @returns
#' No return value.
#' @details
#' Summarizes the object of class \code{iVAE}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'     for (seg in 1:n_segments) {
#'         start_ind <- (seg - 1) * n_per_segment + 1
#'         end_ind <- seg * n_per_segment
#'         latent_data[start_ind:end_ind, i] <- rnorm(
#'             n_per_segment,
#'             runif(1, -5, 5), runif(1, 0.1, 5)
#'         )
#'     }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' summary(res)
#' @export
#' @author Mika Sipilä
#' @method summary iVAE
summary.iVAE <- function(object, ...) {
    cat("Call:", object$call)
    cat("\nData name:", object$DNAME)
    cat("\nDimension of observed data:", object$p)
    cat("\nLatent dimension:", object$call_params$latent_dim)
    cat("\nDimension of auxiliary data:", object$aux_dim)
    cat("\nSample size:", object$n)
    cat("\nSource density model: ", object$call_params$source_dist)
    cat("\nError density model: ", object$call_params$error_dist)
    cat("\nError sigma: ", object$call_params$error_dist_sigma)
}

#' Predict Method for an Object of Class iVAE
#' @description \code{predict} method for the class \code{iVAE}.
#' @param object Object of class \code{iVAE}
#' @param newdata A matrix containing the new data. It can contain
#' new observations or new latent component. Choose the parameter
#' \code{IC_to_data} accordingly.
#' @param aux_data A matrix containing auxiliary data for new observations.
#' Should not be provided if \code{IC_to_data} is \code{TRUE}.
#' @param IC_to_data A boolean defining if the prediction is done from
#' latent components to observations or from observations to latent
#' components. If \code{IC_to_data = TRUE}, \code{newdata} should contain
#' new latent components. Otherwise it should contain the new observations.
#' @param ... Further parameters for \code{predict.default}.
#' @returns
#' A matrix containing the predictions.
#' @details
#' Summarizes the object of class \code{iVAE}.
#' @seealso
#' \code{\link{iVAE}}
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'     for (seg in 1:n_segments) {
#'         start_ind <- (seg - 1) * n_per_segment + 1
#'         end_ind <- seg * n_per_segment
#'         latent_data[start_ind:end_ind, i] <- rnorm(
#'             n_per_segment,
#'             runif(1, -5, 5), runif(1, 0.1, 5)
#'         )
#'     }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' new_data <- matrix(rnorm(p * 2), nrow = 2)
#' new_aux_data <- rbind(c(1, rep(0, n_segments - 1)), c(1, rep(0, n_segments - 1)))
#' pred_ICs <- predict(res, new_data, new_aux_data)
#' new_ICs <- matrix(rnorm(p * 2), nrow = 2)
#' pred_obs <- predict(res, new_ICs, IC_to_data = TRUE)
#' @export
#' @author Mika Sipilä
#' @method predict iVAE
predict.iVAE <- function(
    object, newdata, aux_data = NULL,
    IC_to_data = FALSE, ...) {
    if ("numeric" %in% class(newdata)) {
        newdata <- t(as.matrix(newdata))
    } else {
        newdata <- as.matrix(newdata)
    }
    if (!IC_to_data) {
        if ("numeric" %in% class(aux_data)) {
            aux_data <- t(as.matrix(aux_data))
        } else {
            aux_data <- as.matrix(aux_data)
        }
    }
    dim_in <- ifelse(IC_to_data, object$call_params$latent_dim, object$data_dim)
    n <- dim(newdata)[1]
    p <- dim(newdata)[2]
    if (dim_in != p) {
        stop(paste0(
            "Dimension of newdata does not match the ",
            ifelse(IC_to_data, "latent dimension ",
                "dimension of the data "
            ), "of the fitted model."
        ))
    }
    if (!IC_to_data) {
        aux_dim <- dim(aux_data)[2]
        aux_n <- dim(aux_data)[1]
        if (aux_dim != object$aux_dim) {
            stop("The dimension of the auxiliary data does not match
      the dimension of the auxiliary of the fitted model.")
        }
        if (aux_n != n) {
            stop("The sample size must the same in newdata and aux_data.")
        }
    }
    if (IC_to_data) {
        newdata_unscaled <- sweep(newdata, 2, object$IC_sds, "*")
        newdata_unscaled_uncent <- sweep(newdata_unscaled, 2, object$IC_means, "+")
        obs_data_est <- object$decoder(newdata_unscaled_uncent)
        obs_data_est <- as.data.frame(as.matrix(obs_data_est))
        obs_data_est <- sweep(obs_data_est, 2, object$data_sds, "*")
        obs_data_est <- sweep(obs_data_est, 2, object$data_means, "+")
        names(obs_data_est) <- sapply(1:object$data_dim, FUN = function(x) {
            paste0("X", x)
        })
        return(obs_data_est)
    } else {
        newdata_cent <- sweep(newdata, 2, object$data_means, "-")
        newdata_scaled <- sweep(newdata_cent, 2, object$data_sds, "/")
        IC_est <- object$encoder(list(newdata_scaled, aux_data))
        IC_est <- as.matrix(IC_est)
        IC_est_cent <- sweep(IC_est, 2, object$IC_means, "-")
        IC_est_scaled <- sweep(IC_est_cent, 2, object$IC_sds, "/")
        IC_est_scaled <- as.data.frame(IC_est_scaled)
        names(IC_est_scaled) <- sapply(1:object$call_params$latent_dim,
            FUN = function(x) paste0("IC", x)
        )
        return(IC_est_scaled)
    }
}

#' Save iVAE Object with Trained Tensorflow Models
#' @description Saves \code{iVAE} object including the trained
#' tensorflow models.
#' @inheritDotParams base::save -list
#' @import keras
#' @param object Object of class \code{iVAE}.
#' @param tf_model_dir Directory, where the tensorflow.
#' models should be saved.
#' @param file Filename for saving the \code{iVAE} object.
#' @return
#' No return value.
#' @details
#' The function saves iVAE object correctly by saving also the trained
#' tensorflow models. In a new session, the trained model is available
#' by loading the \code{iVAE} object using the function
#' \code{\link{load_with_tf}}.
#' @seealso
#' \code{\link{load_with_tf}}
#' @author Mika Sipilä
#' @examples
#' p <- 3
#' n_segments <- 10
#' n_per_segment <- 100
#' n <- n_segments * n_per_segment
#' latent_data <- matrix(NA, ncol = p, nrow = n)
#' aux_data <- matrix(0, ncol = n_segments, nrow = n)
#' # Create artificial data with variance and mean varying over the segments.
#' for (i in 1:p) {
#'     for (seg in 1:n_segments) {
#'         start_ind <- (seg - 1) * n_per_segment + 1
#'         end_ind <- seg * n_per_segment
#'         latent_data[start_ind:end_ind, i] <- rnorm(
#'             n_per_segment,
#'             runif(1, -5, 5), runif(1, 0.1, 5)
#'         )
#'     }
#' }
#' mixed_data <- mix_data(latent_data, 2, "elu")
#' # Increase the number of epochs to obtain better performance.
#' res <- iVAE(mixed_data, aux_data, 3, epochs = 10, batch_size = 64)
#' save_with_tf(res, "res_dir", "res_obj.RData")
#' loaded_obj <- load_with_tf("res_obj.RData")
#' new_ICs <- matrix(rnorm(p * 2), nrow = 2)
#' pred_obs <- predict(loaded_obj, new_ICs, IC_to_data = TRUE)
#' @export
save_with_tf <- function(object, tf_model_dir, file, ...) {
    object$tf_model_dir <- tf_model_dir
    save_model_tf(object$encoder, paste0(tf_model_dir, "/encoder"))
    save_model_tf(object$decoder, paste0(tf_model_dir, "/decoder"))
    save_model_tf(object$prior_mean_model, paste0(tf_model_dir, "/prior_mean_model"))
    save(object, file = file, ...)
    print("The model is saved successfully. Use the method load_with_tf to load the model correctly.")
}


#' Load iVAE Object with Trained Tensorflow Models
#' @description Loads \code{iVAE} object including the trained
#' tensorflow models.
#' @import keras
#' @param file Filename of the saved \code{iVAE} object.
#' @return
#' A loaded \code{iVAE} object with restored trained Tensorflow models.
#' @details
#' The function loads iVAE object correctly by loading also the trained
#' tensorflow models.
#' @seealso
#' \code{\link{save_with_tf}}
#' @author Mika Sipilä
#' @inherit save_with_tf examples
#' @export
load_with_tf <- function(file) {
    obj_name <- load(file)
    object <- get(obj_name)
    object$encoder <- load_model_tf(paste0(object$tf_model_dir, "/encoder"))
    object$decoder <- load_model_tf(paste0(object$tf_model_dir, "/decoder"))
    object$prior_mean_model <- load_model_tf(paste0(object$tf_model_dir, "/prior_mean_model"))
    return(object)
}
