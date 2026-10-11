# NonlinearBSS

Nonlinear blind source separation (nonlinear independent component analysis)
with deep neural networks in R.

The observed data are assumed to be a nonlinear mixture `x = f(z)` of
independent latent components `z`. The package estimates the components with

| Function   | Method | Data |
|------------|--------|------|
| `TCL()`    | Time contrastive learning | time series |
| `VAE()`    | Variational autoencoder (not identifiable, a baseline) | i.i.d. data |
| `iVAE()`   | Identifiable variational autoencoder | any data with auxiliary variables: time series, spatial, spatio-temporal |
| `iVAEar()` | iVAE with nonstationary autoregressive components | time series, spatio-temporal and other data with a time component |

## Installation

```r
install.packages("NonlinearBSS")
# or the development version
# remotes::install_github("mikasip/NonlinearBSS")
```

The models are fitted with [keras3](https://keras3.posit.co/) and TensorFlow.
If Python and TensorFlow are not installed yet, run once

```r
keras3::install_keras()
```

and check the setup with `NonlinearBSS::tf_available()`.

## Data formats

All methods accept

- numeric matrices and data frames (coordinates and time given with `coords`
  and `time`, either as values or column names),
- multivariate time series: `ts`, `xts`, `zoo`, `tsibble`,
- spatial and spatio-temporal objects: `sf`, `sftime`, `STFDF` (spacetime).

Coordinates and time points are extracted from the objects, and `components()`,
`predict()` and `fitted()` return results in the same format as the input,
for example an `sf` object with the original geometries.

## Auxiliary data

iVAE type methods need auxiliary variables that explain the nonstationarity of
the latent components. They can be given as a matrix or built from the
coordinates and time points:

| Constructor | Auxiliary variables |
|-------------|---------------------|
| `aux_segments()` | indicators of spatial, temporal or spatio-temporal segments, optional seasons and weekdays |
| `aux_rbf()` | multi-resolution spatial and temporal radial basis functions (the default) |
| `aux_coords()` | normalised coordinates and time |
| `aux_covariates()` | observed covariates |

Several specifications can be combined in a list. With `batch_aux = TRUE`,
radial basis functions are computed batch-wise inside the network, so the
large auxiliary matrix is never stored. `aux_matrix()` computes the auxiliary
data for inspection.

## Example

```r
library(NonlinearBSS)

# 50 stations in Northern Italy, daily data 2020-2022
fit <- iVAEar(pollution_northern_italy,
              aux = aux_rbf(spatial_basis = c(2, 9),
                            temporal_basis = c(9, 17, 37)),
              coords = c("Longitude", "Latitude"), time = "time",
              vars = c("mean_O3", "mean_NO2", "mean_PM10"),
              epochs = 50)
summary(fit)
plot(fit)                               # maps of the components
plot(fit, type = "series", series = 1)  # components at the first station

# trend and variance functions, and forecasts of the components
predict(fit, type = "trend", coords = new_coords, time = new_time)
predict(fit, type = "forecast", coords = new_coords, time = new_time, se = TRUE)

# save and load, including the trained networks
save_nlbss(fit, "fit.rds")
fit <- load_nlbss("fit.rds")
```

Number of components: `select_latent_dim()` fits models with different numbers
of latent components (or `latent_dim_selection()` compares models you fitted
yourself) and selects the one with the largest profile AIC, pAIC = ELBO -
number of components; `plot()` shows the ELBO against the number of
components.

Interpretation: `mshap()` gives the mean absolute SHAP values of the decoder,
i.e. how much each component contributes to each observed variable, and
`plot()` shows them as a heatmap.

Simulation tools: `simulate_sources()` generates nonstationary (optionally
autoregressive) sources, `mix_data()` mixes them nonlinearly and
`absolute_mean_correlation()` measures the separation performance.

## References

- Hyvärinen, A. and Morioka, H. (2016). Unsupervised feature extraction by
  time-contrastive learning and nonlinear ICA. *NeurIPS 29*.
- Khemakhem, I., Kingma, D., Monti, R. and Hyvärinen, A. (2020). Variational
  autoencoders and nonlinear ICA: A unifying framework. *AISTATS*.
- Sipilä, M., Cappello, C., De Iaco, S., Nordhausen, K. and Taskinen, S.
  (2025). Modelling multivariate spatio-temporal data with identifiable
  variational autoencoders. *Neural Networks*, 181, 106774.
- Sipilä, M., Nordhausen, K. and Taskinen, S. (2024). Nonlinear blind source
  separation exploiting spatial nonstationarity. *Information Sciences*, 665,
  120365.
- Sipilä, M., Nordhausen, K. and Taskinen, S. (2025). Identifiable
  autoregressive variational autoencoders for nonlinear and nonstationary
  spatio-temporal blind source separation. In *Machine Learning and Knowledge
  Discovery in Databases, ECML PKDD 2025, Proceedings, Part VII*, 362–380.
  Springer, Cham.
- Sipilä, M., De Iaco, S., Cappello, C., Nordhausen, K., Palma, M. and
  Taskinen, S. (2026). Enhancing identifiable variational autoencoder in the
  presence of missing values and auxiliary covariates for spatio-temporal
  ozone modeling. *Environmetrics*, 37(6), e70121.
- Sipilä, M., Cappello, C., De Iaco, S., Nordhausen, K., Palma, M. and
  Taskinen, S. (2026). Identifiable variational autoencoders for
  interpretable dimension reduction and prediction in multivariate
  spatio-temporal processes. *Spatial Statistics*, 101054.
