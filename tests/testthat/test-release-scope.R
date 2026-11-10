test_that("graph and areal-specific methods are not exported", {
  removed <- c("GiVAE", "givae_logdet", "aux_moran", "aux_laplacian",
               "nlbss_graph", "nlbss_dag", "nlbss_space_graph",
               "nlbss_time_graph", "generate_areal_units", "plot_areal",
               "simulate_sources_areal", "simulate_sources_dag")
  expect_length(intersect(removed, getNamespaceExports("NonlinearBSS")), 0)
  expect_length(intersect(removed, ls(getNamespace("NonlinearBSS"),
                                     all.names = TRUE)), 0)
  expect_true(all(c("TCL", "VAE", "iVAE", "iVAEar", "simulate_sources",
                    "mix_data") %in% getNamespaceExports("NonlinearBSS")))
})
