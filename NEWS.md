# NonlinearBSS 0.2.0

First CRAN release. The package has been rewritten with a unified interface.

## New interface

* Four fitting functions: `TCL()`, `VAE()`, `iVAE()` and `iVAEar()`. They
  accept matrices, data frames, `ts`, `xts`, `zoo`, `tsibble`, `sf`, `sftime`
  and `STFDF` objects, and the results are returned in the input format by
  `components()`, `predict()` and `fitted()`.
* Auxiliary data are specified with `aux_segments()`, `aux_rbf()`,
  `aux_coords()` and `aux_covariates()`, or a list of these, instead of
  separate fitting functions for each type. `aux_matrix()` computes the
  auxiliary data.
* `batch_aux = TRUE` computes all types of auxiliary variables batch-wise
  inside the network, so that the full auxiliary matrix is never stored.
* As in other blind source separation packages (JADE, tsBSS, ICS), the
  estimated components are stored as the element `S` of the fitted object and
  extracted with `components()`.
* `mshap()` computes the (scaled) mean absolute kernel SHAP values of the
  decoder, measuring how much each latent component contributes to each
  observed variable, with `print()` and `plot()` methods.
* `predict()` gives latent components, reconstructions, trends, variances,
  autoregressive coefficients and forecasts with forecast variances. With
  `space = "data"` trends and forecasts are decoded to the observed
  variables, which gives interpolations and forecasts of the data.
* `pAIC()`, `latent_dim_selection()` and `select_latent_dim()` help to
  choose the number of latent components by the profile AIC
  pAIC = ELBO - number of components, with a `plot()` method showing the ELBO
  against the number of components.
* `validation_data` (for example a later time period) and Keras `callbacks`
  (for example early stopping) can be given to all fitting functions.
* Fitted models store their weights, so they survive `saveRDS()`/`readRDS()`;
  `save_nlbss()` and `load_nlbss()` replace `save_with_tf()` and
  `load_with_tf()`.
* `simulate_sources()` replaces the `generate_nonstationary_*()` functions and
  can also simulate autoregressive sources. `mix_data()` now covers
  `mix_data_over_determinated()` (argument `output_dim`) and
  `mix_data_poisson()` (argument `final_activation`).
* Functions with a `seed` argument restore the random number state of the R
  session afterwards.

## Migrating from version 0.1.0

The functions of version 0.1.0 on GitHub are replaced as follows. `x` is the
data, `coords` the spatial coordinates and `time` the time points.

| Version 0.1.0 | Version 0.2.0 |
|---------------|---------------|
| `iVAE(data, aux_data, latent_dim)` | `iVAE(x, aux_data, latent_dim)` |
| `iVAE_coords(data, locations, p)` | `iVAE(x, aux_coords(), p, coords = , time = )` |
| `iVAE_spatial(data, locations, segment_sizes, joint_segment_inds, p)` | `iVAE(x, aux_segments(segment_sizes, joint = joint_segment_inds), p, coords = locations)` |
| `iVAE_radial_spatial(data, locations, p, num_basis, kernel)` | `iVAE(x, aux_rbf(spatial_basis = num_basis, kernel = kernel), p, coords = locations)` |
| `iVAE_radial_spatio_temporal(data, spatial_locations, time_points, p, aux_data = A, ...)` | `iVAE(x, list(aux_rbf(...), aux_covariates(A)), p, coords = , time = )` |
| `iVAE_radial_spatio_temporal_b()`, `iVAE_rbf_b()` | the same with `batch_aux = TRUE` |
| `iVAEar(data, aux_data, p, prev_data_list, prev_aux_data_list)` | `iVAEar(x, aux_data, p, time = , group = )`; the lags are formed automatically |
| `iVAEar_radial(data, spatial_locations, time_points, p, n_s, ...)` | `iVAEar(x, aux_rbf(...), p, coords = , time = )`; `n_s` is not needed |
| `iVAEar_radial_b()` | the same with `batch_aux = TRUE` |
| `iVAEar_segmentation(data, locations, segment_sizes, joint_segment_inds, p, n_s)` | `iVAEar(x, aux_segments(segment_sizes, joint = joint_segment_inds), p, coords = locations[, 1:2], time = locations[, 3])` |
| `fit$IC` | `fit$S` or `components(fit)` |
| `predict(fit, newdata, aux_data)` | `predict(fit, newdata, aux = aux_data)`, or with `coords`/`time` when the auxiliary data are built from them |
| `predict(fit, IC, IC_to_data = TRUE)` | `decode(fit, IC)` |
| `predict_coords_to_IC(fit, coords, time, get_var = FALSE)` | `predict(fit, type = "trend", coords = , time = )` |
| `predict_coords_to_IC(..., get_var = TRUE)` | `predict(fit, type = "variance", coords = , time = )` |
| `predict_coords_to_IC_ar(fit, last_coords, last_time, ..., new_coords, new_time, get_var = TRUE)` | `predict(fit, type = "forecast", coords = new_coords, time = new_time, se = TRUE)`; the last values are taken from the fitted object |
| `predict(fit, predict_coords_to_IC_ar(...)$preds, IC_to_data = TRUE)` | `predict(fit, type = "forecast", coords = , time = , space = "data")` |
| `predict_coords_to_IC_ar_train_data(fit, get_trend, get_var, get_ar_coefs)` | `predict(fit, type = "trend")`, `"variance"` or `"ar_coef"` |
| `save_with_tf(fit, dir, file)`, `load_with_tf(file)` | `save_nlbss(fit, file)`, `load_nlbss(file)`, or `saveRDS()`/`readRDS()` |
| `TCL(data, labels, n_hidden_layers = k, n_hidden_units = m)` | `TCL(x, labels, hidden_units = rep(m, k))` |
| `generate_nonstationary_spatial_data_by_segments(n, p, coords, k)` | `simulate_sources(p, coords = coords, n_segments_space = k)` |
| `generate_nonstationary_spatio_temporal_data_by_segments(n_t, n_s, p, k_s, k_t, coords_time)` | `simulate_sources(p, coords = coords_time[, 1:2], time = coords_time[, 3], n_segments_space = k_s, n_segments_time = k_t, mean_range = c(-1, 1))` |
| `mix_data_over_determinated(data, dim_mixed, n_layers)` | `mix_data(data, n_layers, output_dim = dim_mixed)` |
| `mix_data_poisson(data, n_layers, nonlinearity, final_activation)` | `mix_data(data, n_layers, nonlinearity, final_activation = final_activation)` |

Other changes to be aware of:

* The radial basis function functions of version 0.1.0 standardised the
  additional auxiliary data `aux_data`. In version 0.2.0 this is done by
  `aux_covariates()` (`scale = TRUE` by default); a plain matrix given as `aux`
  is used as it is.
* `get_elbo` is removed; the ELBO is always computed. `add_mask_to_encoder` is
  renamed to `mask_encoder`.
* `TCL()` now returns standardised components in `S`, like the other methods.
* For spatial data only, the Gaussian radial basis functions are no longer
  set to zero beyond distance one, so they are the same as for
  spatio-temporal data.
* `GiVAE()` is not part of this release.

## Bug fixes compared to 0.1.0

* `VAE()` now standardises the data.
* The reparametrisation noise is drawn separately for each observation (it
  was shared within a batch).
* `print()` and `summary()` show the data dimension and sample size.
* Lagged observations of iVAEar are found by series and time point, so the
  data no longer need to be a complete, sorted space-time grid; the ELBO uses
  the correct lag for each order.
