form_aux_data_spatial <- function(locations, segment_sizes, joint_segment_inds = rep(1, length(segment_sizes))) {
    n <- nrow(locations)
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
        labels <- as.numeric(as.factor(labels)) # To ensure that
        # empty segments are reduced
        aux_data <- cbind(aux_data, model.matrix(~ 0 + as.factor(labels)))
    }
    return(aux_data)
}

form_radial_aux_data <- function(spatial_locations, time_points, elevation = NULL, test_inds = NULL, spatial_dim = 2, spatial_basis = c(2, 9), temporal_basis = c(9, 17, 37), elevation_basis = NULL, seasonal_period = NULL, spatial_kernel = "gaussian") {
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
        phi <- cdist(locations_new, knots) / theta
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
    if (!is.null(seasonal_period)) {
        seasons <- floor(time_points / seasonal_period)
        seasons_model_matrix <- model.matrix(~ 0 + as.factor(seasons))
        phi_all <- cbind(phi_all, seasons_model_matrix)
        time_points <- time_points %% seasonal_period + 1
    }
    min_time_point <- min(time_points)
    max_time_point <- max(time_points)
    for (i in seq_along(temporal_basis)) {
        temp_knots <- c(seq(min_time_point, max_time_point, length.out = temporal_basis[i] + 2))
        temp_knots <- temp_knots[2:(length(temp_knots) - 1)]
        temp_dists <- cdist(time_points, temp_knots)
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
            elevation_dists <- cdist(elevation, elevation_knots)
            kappa <- abs(elevation_knots[1] - elevation_knots[2])
            phi <- exp(-0.5*(elevation_dists)^2 / kappa^2)
            phi_all <- cbind(phi_all, phi)
        }
    }
    aux_data <- phi_all
    return(list(aux_data = aux_data, spatial_kernel = spatial_kernel, min_coords = min_coords,
        max_coords = max_coords, min_time_point = min_time_point, max_time_point = max_time_point,
        min_elevation = min_elevation, max_elevation = max_elevation, seasons = seasons))
}
