# The vignette is precomputed because fitting the models needs TensorFlow,
# which is not available on CRAN. After changing NonlinearBSS.Rmd.orig,
# install the package and run this script from the package root:
#   Rscript vignettes/precompile.R
old <- setwd("vignettes")
unlink("figures", recursive = TRUE)
knitr::knit("NonlinearBSS.Rmd.orig", "NonlinearBSS.Rmd")
setwd(old)
