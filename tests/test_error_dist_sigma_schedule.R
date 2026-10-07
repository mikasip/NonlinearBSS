source(file.path("R", "iVAE.R"))

start_sigma <- 0.02
end_sigma <- 0.005

val_0 <- .compute_error_dist_sigma(c(start_sigma, end_sigma), 0, 10000)
val_mid <- .compute_error_dist_sigma(c(start_sigma, end_sigma), 5000, 10000)
val_end <- .compute_error_dist_sigma(c(start_sigma, end_sigma), 10000, 10000)

if (abs(val_0 - start_sigma) > 1e-8) {
  stop("The scheduler should start at the provided starting sigma.")
}
if (abs(val_mid - 0.0125) > 1e-8) {
  stop("The scheduler should interpolate linearly between the start and end values.")
}
if (abs(val_end - end_sigma) > 1e-8) {
  stop("The scheduler should reach the provided ending sigma at the final step.")
}

cat("error_dist_sigma schedule checks passed\n")
