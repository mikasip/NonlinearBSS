# R package for nonlinear blind source separation
The repository contains R package with methods to perform nonlinear blind source separation (BSS). The main methods are variational autoencoder based ```iVAE``` and ```iVAE_spatial```. The package also contains an implementation of regular variational autoencoder (```VAE```) and a method to perform time contrastive learning (```TCL```). 

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
