#' Variational Autoencoder
#'
#' Fits a variational autoencoder (VAE) \insertCite{kingma2013auto}{NonlinearBSS}
#' with a standard normal (or Laplace) prior for the latent components. The
#' model is meant for i.i.d. data.
#'
#' @details
#' The VAE consists of an encoder giving the mean and variance of the
#' approximate posterior \eqn{q(z | x)} and a decoder mapping the latent
#' components back to the data. A sample from the posterior is drawn with the
#' reparametrisation trick and the networks are trained by maximising the
#' evidence lower bound
#' \deqn{E_{q(z|x)}[\log p(x | z) + \log p(z) - \log q(z | x)].}{
#' E_q(z|x) [log p(x|z) + log p(z) - log q(z|x)].}
#'
#' Unlike [iVAE()], a VAE is not identifiable
#' \insertCite{hyvarinen1999nonlinear}{NonlinearBSS}: the latent components are only
#' determined up to an arbitrary invertible transformation and need not
#' correspond to the true sources. The method is mainly useful as a baseline
#' or for dimension reduction.
#'
#' @inheritParams iVAE
#' @return An object of class `c("VAE", "nlbss")`, see [iVAE()].
#' @seealso [iVAE()]
#' @references \insertAllCited{}
#' @examplesIf tf_available()
#' x <- mix_data(matrix(rnorm(3000), ncol = 3), n_layers = 2)
#' fit <- VAE(x, latent_dim = 3, epochs = 5, verbose = 0, seed = 1)
#' fit
#' @export
VAE <- function(x, latent_dim = NULL, vars = NULL,
                hidden_units = c(128, 128, 128), activation = "leaky_relu",
                source_dist = c("gaussian", "laplace"),
                error_dist = c("gaussian", "laplace"),
                error_dist_sigma = 0.01, mask_encoder = TRUE, optimizer = NULL,
                lr_start = 0.01, lr_end = 0.0001, steps = 10000, epochs = 100,
                batch_size = 64, validation_split = 0, validation_data = NULL,
                callbacks = NULL, seed = NULL, verbose = 1) {
  check_common_args(epochs, batch_size, validation_split)
  check_validation(validation_data, validation_split)
  dat <- nlbss_data(x, vars = vars)
  fit_vae_family(
    "VAE", dat, aux = NULL, batch_aux = FALSE, latent_dim, ar_order = 0,
    hidden_units = hidden_units, aux_hidden_units = integer(0),
    activation = activation, source_dist = match.arg(source_dist),
    error_dist = match.arg(error_dist), error_dist_sigma = error_dist_sigma,
    mask_encoder = mask_encoder, optimizer = optimizer, lr_start = lr_start,
    lr_end = lr_end, steps = steps, epochs = epochs, batch_size = batch_size,
    validation_split = validation_split, seed = seed, verbose = verbose,
    call = match.call(), data_name = deparse1(substitute(x)),
    vdat = validation_nlbss(validation_data, dat, NULL, NULL, NULL,
                            character(0)), callbacks = callbacks
  )
}
