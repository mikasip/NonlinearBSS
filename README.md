# R package for nonlinear blind source separation
The repository contains R package with methods to perform nonlinear blind source separation (BSS). The main methods are variational autoencoder based ```iVAE``` and ```iVAEar```. These methods can be called for user specified auxiliary data. Alternatively, for spatial data, the user can directly use the methods ```iVAE_radial_spatial```, ```iVAE_coords``` and ```iVAE_spatial```, and for spatio-temporal data, the methods ```iVAE_radial_spatio_temporal```, ```iVAE_spatial```, ```iVAE_coords```, ```iVAEar_radial``` and ```iVAEar_segmentation``` are available. The methods ```iVAEar_radial``` and ```iVAEar_segmentation``` allow the latent components to have temporal dependence through nonstationary autoregressive processes making them identifiable under nonstationary autoregressive processes or nonstationary variance. The rest of the methods rely on nonstationary variance of the latent components for identifiability.

 # How to install?

Install the package by running R command
```
devtools::install_github("mikasip/NonlinearBSS")
```
The package depends on Tensorflow, which has to be installed in order to run the methods.

To obtain more information about the implemented methods, install the package and read the documentation by running e.g.
```
library(NonlinearBSS)
?iVAE
```
