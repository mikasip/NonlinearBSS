# The vignette is precomputed because fitting the models needs TensorFlow,
# which is not available on CRAN. After changing NonlinearBSS.Rmd.orig,
# install the package and run this script from the package root:
#   Rscript vignettes/precompile.R
old <- setwd("vignettes")
# Use portable Markdown image references for both HTML and PDF output.
knitr::render_markdown()
knitr::opts_chunk$set(dev = "png")
knitr::knit_hooks$set(plot = function(x, options) {
  width <- if (is.null(options$out.width)) "" else
    paste0("{width=", options$out.width, "}")
  paste0("\n\n![](", x, ")", width, "\n\n")
})
knitr::knit("NonlinearBSS.Rmd.orig", "NonlinearBSS.Rmd")
setwd(old)
