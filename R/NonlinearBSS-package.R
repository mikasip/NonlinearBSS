#' NonlinearBSS: Nonlinear Blind Source Separation
#'
#' The package implements deep learning based methods for nonlinear blind
#' source separation (BSS), also known as nonlinear independent component
#' analysis. In all methods the observed \eqn{p}-variate data \eqn{x} are
#' assumed to be generated as \eqn{x = f(z)} where \eqn{f} is an unknown
#' injective (nonlinear) mixing function and \eqn{z} are independent latent
#' components. Without further assumptions the model is not identifiable
#' \insertCite{hyvarinen1999nonlinear}{NonlinearBSS}. The methods differ in
#' how nonstationarity or temporal dependence of the latent components,
#' together with auxiliary variables such as time or location
#' \insertCite{Hyvarinen2019}{NonlinearBSS}, is used to make \eqn{z}
#' identifiable.
#'
#' @section Methods:
#' \describe{
#'   \item{[TCL()]}{Time contrastive learning for time series with
#'   nonstationary variance \insertCite{HyvarinenMorioka2016}{NonlinearBSS}.}
#'   \item{[VAE()]}{A standard variational autoencoder
#'   \insertCite{kingma2013auto}{NonlinearBSS} for i.i.d. data. Not
#'   identifiable, included mainly as a baseline.}
#'   \item{[iVAE()]}{Identifiable variational autoencoder
#'   \insertCite{Khemakhem2020}{NonlinearBSS} for any data with auxiliary
#'   variables, including time series, spatial
#'   \insertCite{sipila2024spatial}{NonlinearBSS} and spatio-temporal data
#'   \insertCite{sipila2024modelling}{NonlinearBSS}.}
#'   \item{[iVAEar()]}{iVAE with nonstationary autoregressive latent
#'   components \insertCite{sipila2025ivaear}{NonlinearBSS} for time series,
#'   spatio-temporal and other data with a time component.}
#' }
#'
#' @section Data formats:
#' All methods accept a numeric matrix or data frame, a multivariate time
#' series (`ts`, `xts`, `zoo`, `tsibble`) or a spatial or spatio-temporal
#' object (`sf`, `sftime`, `STFDF`). Spatial coordinates and time points are
#' extracted automatically, and the estimated components are returned in the
#' same format by [components()][components.nlbss], [predict()][predict.nlbss] and [fitted()][fitted.nlbss].
#'
#' @section Auxiliary data:
#' The auxiliary variables needed by iVAE type methods are either given as a
#' matrix or created from the spatial and temporal locations by
#' [aux_segments()], [aux_rbf()], [aux_coords()] and [aux_covariates()].
#' Several of them can be combined in a list. Radial basis functions can also
#' be computed batch-wise inside the network (`batch_aux = TRUE`), which
#' avoids storing a large auxiliary matrix.
#'
#' @section Interpretation:
#' [mshap()] computes mean absolute SHAP values of the decoder, which show
#' how much each latent component contributes to each observed variable, and
#' [select_latent_dim()] chooses the number of latent components by the
#' profile AIC \insertCite{sipila2026interpretable}{NonlinearBSS}.
#'
#' @section TensorFlow:
#' The models are fitted with \pkg{keras3} using the TensorFlow backend.
#' Python and TensorFlow are installed for example with
#' `keras3::install_keras()`. Use [tf_available()] to check the setup. Fitted
#' models can be saved and loaded with [save_nlbss()] and [load_nlbss()].
#'
#' @references \insertAllCited{}
#' @keywords internal
"_PACKAGE"

#' @importFrom Rdpack reprompt
#' @importFrom stats predict fitted sd runif rnorm cor
#' @importFrom graphics par plot lines legend mtext image axis box
#' @importFrom grDevices hcl.colors
#' @importFrom utils globalVariables
NULL

# `self` and `super` are provided by keras3 inside the methods of custom layers.
utils::globalVariables(c("self", "super"))
