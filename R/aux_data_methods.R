form_aux_data_spatial <- function(locations, segment_sizes, 
    joint_segment_inds = rep(1, length(segment_sizes)),
    time_dim = NULL, seasonal_period = NULL, 
    max_season = NULL, week_component = FALSE) {
    n <- nrow(locations)
    if (week_component) {
        if (is.null(time_dim)) {
            stop("Time dimension must be provided if week component is given.")
        }
        day_of_week <- locations[, time_dim] %% 7
        day_of_week_model_matrix <- model.matrix(~ 0 + as.factor(day_of_week))
    } else {
        day_of_week_model_matrix <- NULL
    }
    if (!is.null(seasonal_period)) {
        if (is.null(time_dim)) {
            stop("Time dimension must be provided if seasonal period is given.")
        }
        seasons <- floor(locations[, time_dim] / seasonal_period)
        if (!is.null(max_season)) {
            seasons <- sapply(seasons, function(i) min(i, max_season))
        }
        seasons_model_matrix <- model.matrix(~ 0 + as.factor(seasons))
        locations[, time_dim] <- locations[, time_dim] %% seasonal_period + 1
    } else {
        seasons <- NULL
        seasons_model_matrix <- NULL
    }
    location_mins <- apply(locations, 2, min)
    locations_zero <- sweep(locations, 2, location_mins, "-")
    location_maxs <- apply(locations_zero, 2, max)
    aux_data <- matrix(nrow = n, ncol = 0)
    
    # Store training labels for each joint segment group
    unique_labels_list <- list()
    
    for (i in unique(joint_segment_inds)) {
        inds <- which(joint_segment_inds == i)
        labels <- rep(0, n)
        lab <- 1
        loop_dim <- function(j, sample_inds) {
            ind <- inds[j]
            seg_size <- segment_sizes[ind]
            seg_limits <- seq(0, (location_maxs[ind]), seg_size)
            for (coord in seg_limits) {
                cur_inds <- which(locations_zero[sample_inds, ind] >= coord &
                    locations_zero[sample_inds, ind] < coord + seg_size)
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
        
        all_possible_labels <- 0:max(labels)  # Include 0 for empty segments
        unique_labels_list[[i]] <- unique(labels)
        
        # Now convert to factor for model matrix creation
        labels_factor <- factor(labels, levels = unique_labels_list[[i]])
        label_model_matrix <- model.matrix(~ 0 + labels_factor)

        aux_data <- cbind(aux_data, label_model_matrix)
    }
    
    if (!is.null(seasons_model_matrix)) {
        aux_data <- cbind(aux_data, seasons_model_matrix)
    }
    if (!is.null(day_of_week_model_matrix)) {
        aux_data <- cbind(aux_data, day_of_week_model_matrix)
    }
    if (!is.null(seasons_model_matrix)) {
        max_season <- max(seasons)
        min_season <- min(seasons)
    } else {
        max_season <- NULL
        min_season <- NULL
    }
    
    return(list(aux_data = aux_data, 
        min_coords = location_mins, 
        max_coords = location_maxs, 
        seasonal_period = seasonal_period, 
        max_season = max_season,
        min_season = min_season,
        seasons = seasons,
        week_component = week_component,
        unique_labels = unique_labels_list))
}

form_radial_params <- function(spatial_locations, time_points, elevation = NULL,
                               spatial_dim = 2, spatial_basis = c(2, 9),
                               temporal_basis = c(9, 17, 37),
                               elevation_basis = NULL, seasonal_period = NULL,
                               max_season = NULL, spatial_kernel = "gaussian", week_component = FALSE) {
  spatial_kernel <- match.arg(spatial_kernel, c("gaussian", "wendland"))
  N <- nrow(spatial_locations)

  min_coords <- apply(spatial_locations, 2, min)
  locations_new <- sweep(spatial_locations, 2, min_coords, "-")
  max_coords <- apply(locations_new, 2, max)

  min_time_point <- min(time_points)
  max_time_point <- max(time_points)

  min_elevation <- if (!is.null(elevation)) min(elevation) else NULL
  max_elevation <- if (!is.null(elevation)) max(elevation) else NULL

  # seasons logic: compute max_season from data if seasonal_period provided and max_season NULL
  seasons <- NULL
  if (!is.null(seasonal_period)) {
    seasons <- floor(time_points / seasonal_period)
    if (!is.null(max_season)) {
      seasons <- sapply(seasons, function(i) min(i, max_season))
    }
    if (is.null(max_season)) max_season <- max(seasons, na.rm = TRUE)
    min_season <- if (!is.null(seasons)) min(seasons, na.rm = TRUE) else NULL
  } else {
    min_season <- NULL
  }

  return(list(
    min_coords = min_coords, max_coords = max_coords,
    min_time_point = min_time_point, max_time_point = max_time_point,
    min_elevation = min_elevation, max_elevation = max_elevation,
    spatial_basis = spatial_basis, temporal_basis = temporal_basis,
    elevation_basis = elevation_basis, spatial_kernel = spatial_kernel,
    week_component = week_component, seasonal_period = seasonal_period,
    max_season = max_season, min_season = min_season
  ))
}

form_radial_aux_data <- function(spatial_locations, time_points, elevation = NULL, 
    spatial_dim = 2, spatial_basis = c(2, 9), temporal_basis = c(9, 17, 37), elevation_basis = NULL, 
    seasonal_period = NULL, max_season = NULL, spatial_kernel = "gaussian", week_component = FALSE) {
    spatial_kernel <- match.arg(spatial_kernel, c("gaussian", "wendland"))
    N <- dim(spatial_locations)[1]
    min_coords <- apply(spatial_locations, 2, min)
    locations_new <- sweep(spatial_locations, 2, min_coords, "-")
    max_coords <- apply(locations_new, 2, max)
    locations_new <- sweep(locations_new, 2, max_coords, "/")
    knots_1d <- sapply(spatial_basis, FUN = function(i) seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i))
    phi_all <- matrix(0, ncol = 0, nrow = N)
    for (i in seq_along(spatial_basis)) {
        theta <- 1 / spatial_basis[i] * 2.5
        knot_list <- replicate(spatial_dim, knots_1d[[i]], simplify = FALSE)
        knots <- as.matrix(expand.grid(knot_list))
        phi <- rdist::cdist(locations_new, knots) / theta
        dist_leq_1 <- phi[which(phi <= 1)]
        dist_g_1_ind <- which(phi > 1)
        if (spatial_kernel == "gaussian") {
            phi <- gaussian_kernel(phi)
        } else {
            phi[which(phi <= 1)] <- wendland_kernel(dist_leq_1)
            phi[dist_g_1_ind] <- 0
        }
        phi_all <- cbind(phi_all, phi)
    }
    seasons <- NULL
    if (week_component) {
        day_of_week <- time_points %% 7
        day_of_week_model_matrix <- model.matrix(~ 0 + as.factor(day_of_week))
        phi_all <- cbind(phi_all, day_of_week_model_matrix)
    }
    if (!is.null(seasonal_period)) {
        seasons <- floor(time_points / seasonal_period)
        if (!is.null(max_season)) {
            seasons <- sapply(seasons, function(i) min(i, max_season))
        }
        seasons_model_matrix <- model.matrix(~ 0 + as.factor(seasons))
        phi_all <- cbind(phi_all, seasons_model_matrix)
        time_points <- time_points %% seasonal_period + 1
    }
    min_time_point <- min(time_points)
    max_time_point <- max(time_points)
    for (i in seq_along(temporal_basis)) {
        temp_knots <- c(seq(min_time_point, max_time_point, length.out = temporal_basis[i] + 2))
        temp_knots <- temp_knots[2:(length(temp_knots) - 1)]
        temp_dists <- rdist::cdist(time_points, temp_knots)
        kappa <- abs(temp_knots[1] - temp_knots[2])
        phi <- exp(-0.5*(temp_dists)^2 / kappa^2)
        phi_all <- cbind(phi_all, phi)
    }
    min_elevation <- NULL
    max_elevation <- NULL
    if (!is.null(elevation)) {
        min_elevation <- min(elevation)
        max_elevation <- max(elevation)
    }
    if (!is.null(elevation) & !is.null(elevation_basis)) {
        for (i in seq_along(elevation_basis)) {
            elevation_knots <- c(seq(min_elevation, max_elevation, length.out = elevation_basis[i] + 2))
            elevation_knots <- elevation_knots[2:(length(elevation_knots) - 1)]
            elevation_dists <- rdist::cdist(elevation, elevation_knots)
            kappa <- abs(elevation_knots[1] - elevation_knots[2])
            phi <- exp(-0.5*(elevation_dists)^2 / kappa^2)
            phi_all <- cbind(phi_all, phi)
        }
    }
    aux_data <- phi_all
    if (!is.null(seasons)) {
        max_season <- max(seasons)
        min_season <- min(seasons)
    } else {
        max_season <- NULL
        min_season <- NULL
    }
    
    return(list(aux_data = aux_data, spatial_kernel = spatial_kernel, min_coords = min_coords,
        max_season = max_season, min_season = min_season,
        max_coords = max_coords, min_time_point = min_time_point, max_time_point = max_time_point,
        min_elevation = min_elevation, max_elevation = max_elevation, seasons = seasons))
}

get_aux_data_radial <- function(object, spatial_locations, time_points,
    elevation = NULL) {
    if (!("iVAEradial_st" %in% class(object))) {
        stop("Object must be class of iVAEradial_st")
    }
    N <- dim(spatial_locations)[1]
    locations_new <- sweep(spatial_locations, 2, object$min_coords, "-")
    locations_new <- sweep(locations_new, 2, object$max_coords, "/")
    knots_1d <- sapply(object$spatial_basis, FUN = function(i) {
        seq(0 + (1 / (i + 2)), 1 - (1 / (i + 2)), length.out = i)
    })
    phi_all <- matrix(0, ncol = 0, nrow = N)
    for (i in seq_along(object$spatial_basis)) {
        theta <- 1 / object$spatial_basis[i] * 2.5
        knot_list <- replicate(object$spatial_dim, knots_1d[[i]], simplify = FALSE)
        knots <- as.matrix(expand.grid(knot_list))
        phi <- rdist::cdist(locations_new, knots) / theta
        dist_leq_1 <- phi[which(phi <= 1)]
        dist_g_1_ind <- which(phi > 1)
        if (object$spatial_kernel == "gaussian") {
            phi <- gaussian_kernel(phi)
        } else {
            phi[which(phi <= 1)] <- wendland_kernel(dist_leq_1)
            phi[dist_g_1_ind] <- 0
        }
        phi_all <- cbind(phi_all, phi)
    }
    if (object$week_component) {
        day_of_week <- time_points %% 7
        day_of_week_model_matrix <- model.matrix(~ 0 + as.factor(day_of_week))
        phi_all <- cbind(phi_all, day_of_week_model_matrix)
    }
    if (!is.null(object$seasonal_period)) {
        seasons <- c(
            object$min_season:object$max_season,
            floor(time_points / object$seasonal_period)
        )
        seasons[seasons > object$max_season] <- object$max_season
        seasons_model_matrix <- model.matrix(~ 0 + as.factor(seasons))
        phi_all <- cbind(
            phi_all,
            seasons_model_matrix[-c(1:((object$max_season - object$min_season + 1))), ]
        )
        time_points <- time_points %% object$seasonal_period + 1
    }
    for (i in seq_along(object$temporal_basis)) {
        temp_knots <- c(seq(object$min_time_point, object$max_time_point,
            length.out = object$temporal_basis[i] + 2
        ))
        temp_knots <- temp_knots[2:(length(temp_knots) - 1)]
        temp_dists <- rdist::cdist(time_points, temp_knots)
        kappa <- abs(temp_knots[1] - temp_knots[2])
        phi <- exp(-0.5 * (temp_dists)^2 / kappa^2)
        phi_all <- cbind(phi_all, phi)
    }
    if (!is.null(elevation)) {
        for (i in seq_along(object$elevation_basis)) {
            elevation_knots <- c(seq(object$min_elevation, object$max_elevation,
                length.out = object$elevation_basis[i] + 2
            ))
            elevation_knots <- elevation_knots[2:(length(elevation_knots) - 1)]
            elevation_dists <- rdist::cdist(elevation, elevation_knots)
            kappa <- abs(elevation_knots[1] - elevation_knots[2])
            phi <- exp(-0.5 * (elevation_dists)^2 / kappa^2)
            phi_all <- cbind(phi_all, phi)
        }
    }
    return(phi_all)
}

get_aux_data_spatial <- function(object, locations) {
    if (!("iVAEar_segmentation" %in% class(object))) {
        stop("Object must be class of iVAEar_segmentation")
    }
    N <- nrow(locations)
    
    # Handle week component
    if (object$week_component) {
        day_of_week <- locations[, object$time_dim] %% 7
        day_of_week_model_matrix <- model.matrix(~ 0 + as.factor(day_of_week))
        # Ensure all 7 days are represented in columns
        all_days <- 0:6
        day_names <- paste0("as.factor(day_of_week)", all_days)
        missing_days <- setdiff(day_names, colnames(day_of_week_model_matrix))
        if (length(missing_days) > 0) {
            missing_matrix <- matrix(0, nrow = N, ncol = length(missing_days))
            colnames(missing_matrix) <- missing_days
            day_of_week_model_matrix <- cbind(day_of_week_model_matrix, missing_matrix)
            # Reorder columns to match expected order
            day_of_week_model_matrix <- day_of_week_model_matrix[, day_names]
        }
    } else {
        day_of_week_model_matrix <- NULL
    }
    
    # Handle seasonal component
    if (!is.null(object$seasonal_period)) {
        seasons <- floor(locations[, object$time_dim] / object$seasonal_period)
        seasons[seasons > object$max_season] <- object$max_season
        
        # Create model matrix with all possible seasons from training
        all_seasons <- 0:object$max_season
        season_names <- paste0("as.factor(seasons)", all_seasons)
        
        # Create full model matrix
        seasons_expanded <- factor(seasons, levels = all_seasons)
        seasons_model_matrix <- model.matrix(~ 0 + seasons_expanded)
        colnames(seasons_model_matrix) <- season_names
        
        # Modify time dimension
        locations[, object$time_dim] <- locations[, object$time_dim] %% 
            object$seasonal_period + 1
    } else {
        seasons_model_matrix <- NULL
    }
    
    locations_zero <- sweep(locations, 2, object$min_coords, "-")
    aux_data <- matrix(nrow = N, ncol = 0)
    
    # Process each joint segment group
    for (i in unique(object$joint_segment_inds)) {
        inds <- which(object$joint_segment_inds == i)
        labels <- rep(0, N)
        lab <- 1
        
        loop_dim <- function(j, sample_inds) {
            ind <- inds[j]
            seg_size <- object$segment_sizes[ind]
            seg_limits <- seq(0, (object$max_coords[ind]), seg_size)
            for (coord in seg_limits) {
                cur_inds <- which(locations_zero[sample_inds, ind] >= coord &
                    locations_zero[sample_inds, ind] < coord + seg_size)
                cur_sample_inds <- sample_inds[cur_inds]
                if (j == length(inds)) {
                    labels[cur_sample_inds] <<- lab
                    lab <<- lab + 1
                } else {
                    loop_dim(j + 1, cur_sample_inds)
                }
            }
        }
        loop_dim(1, 1:N)
        
        # Get the training labels for this joint segment group
        unique_new_labels <- unique(labels)
        training_labels <- object$unique_labels[[i]]
        if (any(!unique_new_labels %in% training_labels)) {
            stop("The segments of the new data do not match the training data")
        }
        labels_factor <- factor(labels, levels = training_labels)
        
        label_model_matrix <- model.matrix(~ 0 + labels_factor)
        
        aux_data <- cbind(aux_data, label_model_matrix)
    }
    
    # Add seasonal and week components
    if (!is.null(seasons_model_matrix)) {
        aux_data <- cbind(aux_data, seasons_model_matrix)
    }
    if (!is.null(day_of_week_model_matrix)) {
        aux_data <- cbind(aux_data, day_of_week_model_matrix)
    }
    
    return(aux_data)
}


layer_radial_basis <- keras3::Layer(
  "RadialBasisLayer",
  initialize = function(spatial_basis, temporal_basis,
                        min_coords, max_coords,
                        min_time_point, max_time_point,
                        spatial_kernel = "wendland",
                        week_component = FALSE,
                        seasonal_period = NULL,
                        max_season = NULL,
                        min_season = NULL,
                        elevation_basis = NULL,
                        min_elevation = NULL,
                        max_elevation = NULL) {
    super$initialize()
    self$spatial_basis <- as.integer(spatial_basis)
    self$temporal_basis <- as.integer(temporal_basis)
    self$min_coords <- tensorflow::tf$constant(min_coords, dtype = "float32")
    self$max_coords <- tensorflow::tf$constant(max_coords, dtype = "float32")
    self$spatial_kernel <- spatial_kernel
    self$spatial_dim <- as.integer(length(min_coords))

    self$seasonal_period <- if (!is.null(seasonal_period)) as.integer(seasonal_period) else NULL
    # FIXED: Set effective min/max to match baseline behavior after modulo
    if (!is.null(self$seasonal_period)) {
      min_time_point_eff <- 1.0
      max_time_point_eff <- as.numeric(self$seasonal_period)
    } else {
      min_time_point_eff <- as.numeric(min_time_point)
      max_time_point_eff <- as.numeric(max_time_point)
    }
    self$min_time_point <- tensorflow::tf$constant(min_time_point_eff, dtype = "float32")
    self$max_time_point <- tensorflow::tf$constant(max_time_point_eff, dtype = "float32")

    # Precompute spatial knots + thetas
    self$spatial_knots <- list()
    thetas <- numeric(length(spatial_basis))
    for (i in seq_along(spatial_basis)) {
      knots_1d <- seq(0 + (1 / (spatial_basis[i] + 2)),
                      1 - (1 / (spatial_basis[i] + 2)),
                      length.out = spatial_basis[i])
      knot_arrays <- replicate(self$spatial_dim, knots_1d, simplify = FALSE)
      knots_grid <- as.matrix(expand.grid(knot_arrays))
      self$spatial_knots[[i]] <- tensorflow::tf$constant(knots_grid, dtype = "float32")
      thetas[i] <- 1 / spatial_basis[i] * 2.5
    }
    self$spatial_thetas <- tensorflow::tf$constant(thetas, dtype = "float32")

    # Precompute temporal knots + kappas
    self$temporal_knots <- list()
    kappas <- numeric(length(temporal_basis))
    for (i in seq_along(temporal_basis)) {
      temp_knots <- seq(min_time_point_eff, max_time_point_eff, length.out = temporal_basis[i] + 2)
      temp_knots <- temp_knots[2:(length(temp_knots) - 1)]
      self$temporal_knots[[i]] <- tensorflow::tf$constant(matrix(temp_knots, ncol = 1), dtype = "float32")
      kappa <- abs(temp_knots[1] - temp_knots[2])
      kappas[i] <- kappa^2
    }
    self$temporal_kappas <- tensorflow::tf$constant(kappas, dtype = "float32")

    # Elevation params
    self$elevation_basis <- if (!is.null(elevation_basis)) as.integer(elevation_basis) else NULL
    if (!is.null(min_elevation)) self$min_elevation <- tensorflow::tf$constant(min_elevation, dtype = "float32") else self$min_elevation <- NULL
    if (!is.null(max_elevation)) self$max_elevation <- tensorflow::tf$constant(max_elevation, dtype = "float32") else self$max_elevation <- NULL

    # FIXED: Store unique season indices for proper one-hot encoding
    self$week_component <- week_component
    self$max_season <- if (!is.null(max_season)) as.integer(max_season) else NULL
    self$min_season <- if (!is.null(min_season)) as.integer(min_season) else 0L
    
    # For one-hot encoding, we need the actual number of unique seasons
    # This should match what model.matrix would create
    if (!is.null(self$seasonal_period) && !is.null(self$max_season)) {
      # Compute how many unique seasons exist in the range
      self$n_seasons <- as.integer(self$max_season - self$min_season + 1L)
    } else {
      self$n_seasons <- NULL
    }
  },

  call = function(inputs, mask = NULL) {
    spatial_locations <- inputs[[1]]
    time_points <- inputs[[2]]
    has_elevation <- length(inputs) >= 3
    elevation <- if (has_elevation) inputs[[3]] else NULL

    # normalize spatial
    locations_normalized <- (spatial_locations - self$min_coords) / self$max_coords

    phi_list <- list()

    # Spatial RBFs (these are correct!)
    for (i in seq_along(self$spatial_basis)) {
      locs_expanded <- tensorflow::tf$expand_dims(locations_normalized, axis = 2L)
      knots_expanded <- tensorflow::tf$expand_dims(self$spatial_knots[[i]], axis = 0L)
      knots_expanded <- tensorflow::tf$transpose(knots_expanded, perm = c(0L, 2L, 1L))
      diff_sq <- (locs_expanded - knots_expanded)^2
      distances <- tensorflow::tf$sqrt(tensorflow::tf$reduce_sum(diff_sq, axis = 1L))
      phi <- distances / self$spatial_thetas[i]

      if (self$spatial_kernel == "gaussian") {
        phi <- tensorflow::tf$exp(- (phi^2))
      } else {
        inside <- phi <= 1.0
        w <- ((1 - phi)^6) * ((35 * (phi^2) + 18 * phi + 3) / 3)
        phi <- tensorflow::tf$where(inside, w, tensorflow::tf$zeros_like(w))
      }
      phi_list[[length(phi_list) + 1]] <- phi
    }

    if (self$week_component) {
      dow <- tensorflow::tf$math$mod(tensorflow::tf$cast(time_points, "int32"), 7L)
      one_hot_week <- tensorflow::tf$one_hot(dow, 7L)
      one_hot_week <- tensorflow::tf$reshape(one_hot_week, shape = c(-1L, 7L))
      phi_list[[length(phi_list) + 1]] <- tensorflow::tf$cast(one_hot_week, "float32")
    }

    # FIXED: Seasonal encoding with proper depth matching baseline
    if (!is.null(self$seasonal_period)) {
      seasons_raw <- tensorflow::tf$math$floordiv(
        tensorflow::tf$cast(time_points, "int32"), 
        as.integer(self$seasonal_period)
      )
      
      # Cap at max_season if provided
      if (!is.null(self$max_season)) {
        seasons_raw <- tensorflow::tf$minimum(seasons_raw, as.integer(self$max_season))
      }
      
      # FIXED: Adjust to start from 0 for one-hot encoding
      seasons_adj <- seasons_raw - as.integer(self$min_season)
      
      # FIXED: Use n_seasons for depth to match model.matrix behavior
      depth <- if (!is.null(self$n_seasons)) self$n_seasons else 1L
      season_onehot <- tensorflow::tf$one_hot(seasons_adj, depth)
      season_onehot <- tensorflow::tf$reshape(season_onehot, shape = c(-1L, depth))
      phi_list[[length(phi_list) + 1]] <- tensorflow::tf$cast(season_onehot, "float32")

      # FIXED: Add 1 to match baseline (range [1, seasonal_period] instead of [0, seasonal_period-1])
      time_for_temporal <- tensorflow::tf$cast(
        tensorflow::tf$math$mod(
          tensorflow::tf$cast(time_points, "int32"), 
          as.integer(self$seasonal_period)
        ), 
        "float32"
      ) + 1.0  # <-- THIS IS THE KEY FIX!
    } else {
      time_for_temporal <- tensorflow::tf$cast(time_points, "float32")
    }

    # Temporal RBFs (now with corrected time range)
    for (i in seq_along(self$temporal_basis)) {
      time_expanded <- tensorflow::tf$expand_dims(time_for_temporal, axis = -1L)
      temp_knots_expanded <- tensorflow::tf$expand_dims(self$temporal_knots[[i]], axis = 0L)
      temp_knots_expanded <- tensorflow::tf$transpose(temp_knots_expanded, perm = c(0L, 2L, 1L))
      temp_dist <- tensorflow::tf$abs(time_expanded - temp_knots_expanded)
      temp_dist <- tensorflow::tf$squeeze(temp_dist, axis = 1L)
      phi_temp <- tensorflow::tf$exp(-0.5 * (temp_dist^2) / self$temporal_kappas[i])
      phi_list[[length(phi_list) + 1]] <- phi_temp
    }

    # Elevation RBFs (these look correct)
    if (!is.null(self$elevation_basis) && !is.null(elevation) && !is.null(self$min_elevation) && !is.null(self$max_elevation)) {
      elev_norm <- (elevation - self$min_elevation) / (self$max_elevation - self$min_elevation)
      for (i in seq_along(self$elevation_basis)) {
        elev_knots <- seq(0, 1, length.out = self$elevation_basis[i] + 2)
        elev_knots <- elev_knots[2:(length(elev_knots) - 1)]
        knots_t <- tensorflow::tf$constant(matrix(elev_knots, ncol = 1), dtype = "float32")
        elev_exp <- tensorflow::tf$expand_dims(elev_norm, axis = -1L)
        dist <- tensorflow::tf$abs(elev_exp - knots_t)
        kappa <- abs(elev_knots[1] - elev_knots[2])
        phi_e <- tensorflow::tf$exp(-0.5 * dist^2 / (kappa^2))
        phi_list[[length(phi_list) + 1]] <- phi_e
      }
    }

    phi_all <- tensorflow::tf$concat(phi_list, axis = 1L)
    return(phi_all)
  }
)
