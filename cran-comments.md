## Submission

This is the first CRAN submission of NonlinearBSS.

## Test environments

* local Windows 11, R-devel (2026-07-24 r90297 ucrt), with TensorFlow 2.21

## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## TensorFlow

The model fitting functions need Python with TensorFlow, which is not
available on CRAN machines. Examples that fit models are therefore run only
when `tf_available()` is `TRUE`. This check never initialises Python (and so
never triggers an automatic Python installation) in non-interactive sessions,
unless the environment variable `NOT_CRAN` is `"true"`. Tests of the models
are skipped on CRAN with `skip_on_cran()`; the other functions (data
conversion, auxiliary data, simulation) are tested and have running examples.

The vignette is precomputed locally with TensorFlow (the source is
`vignettes/NonlinearBSS.Rmd.orig`, excluded from the build), so it contains
no code that is evaluated on CRAN.
