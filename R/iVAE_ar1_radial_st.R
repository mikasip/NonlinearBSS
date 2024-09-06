iVAE_spatio_temporal_ar1 <- function(data, locations, n_s, segment_sizes, joint_segment_inds = rep(1, length(segment_sizes)), latent_dim, test_inds = NULL, epochs, batch_size, ...) {
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
        cur_inds <- which(locations[sample_inds, ind] >= coord & locations[sample_inds, ind] < coord + seg_size)
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
    labels <- as.numeric(as.factor(labels)) # To ensure that empty segments are reduced
    aux_data <- cbind(aux_data, model.matrix(~ 0 + as.factor(labels)))
  }
  test_data <- NULL
  if (!is.null(test_inds)) {
    test_data <- data[test_inds, ]
    test_aux_data <- aux_data[test_inds, ]
  }
  data_prev <- rbind(data[1:n_s, ], data[1:(n - n_s), ])
  resVAE <- iVAE_ar1(data, aux_data, latent_dim, data_prev = data_prev, test_data = test_data, test_data_aux = test_aux_data, epochs = epochs, batch_size = batch_size, ...)
  class(resVAE) <- c("iVAEspatial", class(resVAE))
  resVAE$spatial_dim <- dim(locations)[2]
  return(resVAE)
}
