% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/TCL.R
\name{TCL}
\alias{TCL}
\title{Time Contrastive Learning}
\usage{
TCL(
  data,
  labels,
  n_hidden_layers = 1,
  n_hidden_units = 32,
  lr_start = 0.01,
  lr_end = 0.001,
  seed = NULL,
  ...
)
}
\arguments{
\item{data}{A matrix with P columns and N rows containing the observed data.}

\item{labels}{A vector of length N containing the labels
for the observations.}

\item{n_hidden_layers}{A number of hidden layers in TCL model.}

\item{n_hidden_units}{A number of hidden units in each hidden layer.}

\item{lr_start}{A starting learning rate.}

\item{lr_end}{A learning rate after polynomial decay.}

\item{seed}{Seed for the tensorflow model.}

\item{...}{Further parameters for keras::keras3::fit function.}
}
\value{
An object of class TCL.
}
\description{
Constructs and keras3::fits a model for time contrastive learning.
}
\details{
The method constructs and keras3::fits a model for time contrastive learning
based on the given parameters. TCL assigns half of the labels incorrectly to
observations, and trains a deep neural network model to predict if the label
is correct or incorrect. The model has a bottleneck layer at the end, which
has P hidden units. The idea is, that in order to classify the data
correctly, it has to learn the true independent components. For more
details about TCL, see \insertCite{HyvarinenMorioka2016}{NonlinearBSS}.
}
\examples{
p <- 10
n_segments <- 100
n_per_segment <- 50
n <- n_segments * n_per_segment
latent_data <- matrix(NA, ncol = p, nrow = n)
labels <- numeric(n)
# Create artificial data with variance and mean varying over the segments.
for (seg in 1:n_segments) {
    start_ind <- (seg - 1) * n_per_segment + 1
    end_ind <- seg * n_per_segment
    labels[start_ind:end_ind] <- seg
    for (i in 1:p) {
        latent_data[start_ind:end_ind, i] <- rnorm(
            n_per_segment,
            0, runif(1, 0.1, 5)
        )
    }
}
mixed_data <- mix_data(latent_data, 2, "elu")

# For better performance, increase the number of epochs.
res <- TCL(mixed_data, labels - 1,
    n_hidden_layers = 1,
    n_hidden_units = 32, batch_size = 64, epochs = 10
)
cormat <- cor(res$IC, latent_data)
cormat
absolute_mean_correlation(cormat)

}
\references{
\insertAllCited{}
}
\author{
Mika Sipilä
}
