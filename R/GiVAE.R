#' Graph Identifiable Variational Autoencoder
#'
#' Fits an identifiable variational autoencoder whose latent components depend
#' on their neighbours in one or several graphs (GiVAE). It is meant for areal
#' and network data, spatio-temporal data on areal units, and any data where
#' the dependence between observations is described by graphs.
#'
#' @details
#' For observation \eqn{i} the prior of the latent components is
#' \deqn{z_{il} | \{\hat z_j\}, u_i \sim N(\mu_{il}, \sigma_l^2(u_i)),\quad
#' \mu_{il} = \rho_{max}\,\rho_l(u_i) \sum_{g=1}^G \beta_{gl}(u_i)
#' \frac{1}{|\partial_g i|}\sum_{j \in \partial_g i} \hat z_{jl},}{
#' z_il ~ N(mu_il, sigma_l^2(u_i)), mu_il = rho_max rho_l(u_i) sum_g
#' beta_gl(u_i) mean_{j in N_g(i)} zhat_jl,}
#' where \eqn{\hat z_j} is the encoder mean of neighbour \eqn{j},
#' \eqn{\rho_l(u)} in \eqn{(0, 1)} (scaled by `rho_max`) is the total
#' coupling strength of component \eqn{l}, \eqn{\beta_{gl}(u)} (softmax over
#' the graphs) distributes it between the graphs, and \eqn{\sigma_l^2(u)} is
#' the nonstationary prior variance. The coupling and the variance are
#' component-specific; this is what makes the components identifiable from
#' nonstationary dependence (with a coupling shared by all components the
#' prior is invariant to rotations of the latent space when the variances are
#' equal). The neural networks for \eqn{\rho}, \eqn{\beta} and
#' \eqn{\sigma^2} take the auxiliary variables \eqn{u} as input.
#'
#' The graphs are given as [nlbss_graph()] objects. A graph can have fewer
#' nodes than the data have rows, for example a spatial graph of `n_s` areas
#' for `n_s * n_t` rows; the layout of each graph tells how its nodes are
#' repeated over the rows. Neighbours of a row are always searched in the
#' same position of the other dimensions (for a spatial graph: the same time
#' point).
#'
#' By default the auxiliary variables are Laplacian eigenvectors of each graph
#' ([aux_laplacian()]). They are computed from the whole graph and not
#' batch-wise.
#'
#' The prior uses the neighbours' encoder means in a conditional (pseudo
#' likelihood) form; the determinant term of a joint simultaneous
#' autoregressive model is not included. For a directed graph that only points
#' backwards in time the determinant is 1 and the prior is an exact
#' autoregressive model; for undirected (e.g. spatial) graphs the omission
#' favours smooth latent fields and strong coupling.
#'
#' @inheritParams iVAE
#' @param graphs A graph specification [nlbss_graph()], a list of them, a
#'   matrix, or an adjacency list. See Details.
#' @param aux Auxiliary data. Defaults to [aux_laplacian()] of `graphs` with
#'   `K` eigenvectors per graph. May also be a matrix, a list of
#'   specifications, or combined with them.
#' @param K Number of Laplacian eigenvectors per graph if `aux` is not given.
#' @param rho_max Upper bound of the total coupling strength, in (0, 1].
#' @param jacobian Log-determinant term of the simultaneous autoregressive
#'   prior: `"none"` (default; the objective then omits \eqn{\log|\det(I -
#'   B)|} and is exact only for acyclic graphs such as a directed time graph),
#'   `"exact"` (dense log-determinant of the sampled component, for components
#'   up to `jacobian_max_exact` nodes) or `"stochastic"` (truncated series with
#'   Hutchinson probes, for any size). The term is estimated unbiasedly by
#'   sampling one strongly connected component of the observation graph per
#'   training step, e.g. one time point for a spatial graph. It needs the
#'   coupling networks to be evaluated on that component at every step, so it
#'   is slow for one very large graph. Pure acyclic graphs need no term and
#'   `jacobian` is then ignored. See [givae_logdet()] for evaluating it exactly
#'   after fitting.
#' @param jacobian_terms,jacobian_probes Number of series terms and probe
#'   vectors for `jacobian = "stochastic"`. The series is truncated, which
#'   under-estimates the penalty when `rho_max` is close to 1; the cost and
#'   memory grow with both.
#' @param jacobian_max_exact Largest component size allowed for `"exact"`.
#' @param max_neighbors Optional cap on the number of neighbours per graph and
#'   row (a fixed random subset is used for nodes with more neighbours). Saves
#'   memory for graphs with hubs.
#' @return An object of class `c("GiVAE", "iVAE", "nlbss")`, see [iVAE()].
#'   Additional elements: `graphs`, `neighbors` (the row indices of the
#'   neighbours) and `rho_max`.
#' @seealso [nlbss_graph()], [aux_laplacian()], [iVAE()]
#' @export
GiVAE <- function(x, graphs, aux = NULL, K = 10L, latent_dim = NULL,
                  coords = NULL, time = NULL, group = NULL, vars = NULL,
                  rho_max = 0.99, max_neighbors = NULL,
                  jacobian = c("none", "exact", "stochastic"),
                  jacobian_terms = 30L, jacobian_probes = 2L,
                  jacobian_max_exact = 1500L,
                  hidden_units = c(128, 128, 128),
                  aux_hidden_units = c(128, 128, 128),
                  activation = "leaky_relu",
                  source_dist = c("gaussian", "laplace"),
                  error_dist = c("gaussian", "laplace"),
                  error_dist_sigma = 0.02, mask_encoder = TRUE,
                  optimizer = NULL, lr_start = 0.001, lr_end = 0.0001,
                  steps = 10000, epochs = 100, batch_size = 64,
                  validation_split = 0, callbacks = NULL, seed = NULL,
                  verbose = 1) {
  check_common_args(epochs, batch_size, validation_split)
  jacobian <- match.arg(jacobian)
  if (!is.numeric(rho_max) || length(rho_max) != 1 || rho_max <= 0 || rho_max > 1) {
    stop("`rho_max` must be in (0, 1].", call. = FALSE)
  }
  graphs <- as_graph_list(graphs)
  if (is.null(aux)) aux <- aux_laplacian(graphs, K = K)
  dat <- nlbss_data(x, coords = coords, time = time, group = group,
                    vars = vars, aux_cols = aux_columns(aux))
  fit_vae_family(
    "GiVAE", dat, aux, batch_aux = FALSE, latent_dim = latent_dim, ar_order = 0,
    hidden_units = hidden_units, aux_hidden_units = aux_hidden_units,
    activation = activation, source_dist = match.arg(source_dist),
    error_dist = match.arg(error_dist), error_dist_sigma = error_dist_sigma,
    mask_encoder = mask_encoder, optimizer = optimizer, lr_start = lr_start,
    lr_end = lr_end, steps = steps, epochs = epochs, batch_size = batch_size,
    validation_split = validation_split, seed = seed, verbose = verbose,
    call = match.call(), data_name = deparse1(substitute(x)),
    vdat = NULL, callbacks = callbacks,
    graphs = graphs, rho_max = rho_max, max_neighbors = max_neighbors,
    jacobian = jacobian, jacobian_terms = as.integer(jacobian_terms),
    jacobian_probes = as.integer(jacobian_probes),
    jacobian_max_exact = as.integer(jacobian_max_exact)
  )
}