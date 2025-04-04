% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/iVAEar_segmentation.R
\name{iVAEar_segmentation}
\alias{iVAEar_segmentation}
\title{iVAEar Segmentation}
\usage{
iVAEar_segmentation(
  data,
  locations,
  segment_sizes,
  joint_segment_inds = rep(1, length(segment_sizes)),
  latent_dim,
  n_s,
  ar_order = 1,
  epochs,
  batch_size,
  ...
)
}
\arguments{
\item{data}{A matrix of observed data (n x p) where each row is an observation and each column is a feature.}

\item{locations}{A matrix of spatio-temporal locations corresponding to each observation in \code{data}.}

\item{segment_sizes}{A vector providing sizes for segments.
The dimension should match the spatial dimenstion.}

\item{joint_segment_inds}{A vector indicating which segments
are considered jointly. See more in details.}

\item{latent_dim}{A latent dimension for iVAE.}

\item{n_s}{Number of unique spatial locations in the data.}

\item{ar_order}{An autoregressive order used in iVAEar.}

\item{epochs}{Integer specifying the number of training epochs for the iVAE model.}

\item{batch_size}{Integer specifying the batch size for training the iVAE model.}

\item{...}{Additional arguments passed to the underlying \code{\link{iVAEar}} function.}
}
\value{
An object of class iVAEspatial, inherits from class iVAE.
Additionally, the object has a property
\code{spatial_dim} which gives the dimension of the given locations.
For more details, see \code{\link{iVAE}}.
}
\description{
Fits an identifiable Variational Autoencoder with an autoregressive prior to segmented data. 
This function is designed for spatio-temporal data, where the data is divided into 
segments based on their spatio-temporal locations. The segmentation is used as auxiliary data.
}
\details{
The method creates the auxiliary data as spatio-temporal segments based on
the given input parameters.
The vector \code{segment_sizes} defines the size of the segments for
each spatial/temporal dimension separately. The segmentation is then created
based on the \code{joint_segment_inds}, which defines dimensions are
considered jointly. For example for spatio-temporal data with two spatial
dimensions, \code{joint_segment_inds = c(1, 1, 2)}
defines that the spatial dimensions are
considered jointly, and the temporal dimension is considered alone. 
This means that the auxiliary variable is vector
giving the two dimensional spatial segment and one dimensional temporal
segment in which the observation belongs in. All dimensions are considered jointly as
default.

After the segmentation, the method calls the function \code{iVAEar}
using the created auxiliary variables.
}
\examples{
p <- 3
n_time <- 100
n_spat <- 50
coords_time <- cbind(
    rep(runif(n_spat), n_time), rep(runif(n_spat), n_time),
    rep(1:n_time, each = n_spat)
)
data_obj <- generate_nonstationary_spatio_temporal_data_by_segments(
    n_time,
    n_spat, p, 5, 10, coords_time
)
latent_data <- data_obj$data
# Generate artificial observed data by applying a nonlinear mixture
obs_data <- mix_data(latent_data, 2)

# Increase the number of epochs for better performance.
resiVAE <- iVAEar_segmentation(
  data = obs_data, 
  locations = coords_time, 
  segment_sizes = c(0.1, 0.1, 5), 
  joint_segment_inds = c(1, 1, 2),
  latent_dim = p, 
  n_s = n_spat,
  epochs = 1,
  batch_size = 64
)
cormat <- cor(resiVAE$IC, latent_data)
cormat
absolute_mean_correlation(cormat)

}
\seealso{
\code{\link{iVAEar}}
}
\author{
Mika Sipilä
}
