library(BayesSpace)
library(dplyr)
section_list <- c("151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676")

for(i in section_list) {
  dlpfc <- getRDS("2020_maynard_prefrontal-cortex", i)

  set.seed(101)
  dec <- scran::modelGeneVar(dlpfc)
  top <- scran::getTopHVGs(dec, n = 2000)
  
  set.seed(102)
  dlpfc <- scater::runPCA(dlpfc, subset_row=top)
  
  dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)
  
  if (i %in% c('151669', '151670', '151671', '151672')) {
    q <- 5
  } else {
    q <- 7
  }
  d <- 15  # Number of PCs
  
  set.seed(104)
  dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium',
                          nrep=50000, gamma=3, save.chain=TRUE)
  
  if (i %in% c('151669', '151670', '151671', '151672')) {
    labels <- dplyr::recode(dlpfc$spatial.cluster, 0, 1, 2, 3, 4)
  } else {
    labels <- dplyr::recode(dlpfc$spatial.cluster, 0, 1, 2, 3, 4, 5, 6)
  }
  write.csv(labels, paste("dlpfc", paste(i, "txt", sep="."), sep="_"))
}