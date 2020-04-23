tf_enrichment <- function(sigs, cmap_path, landmark){
  #landmark = landmark annotation
  #sigs = vector of signature IDs
  #cmap_path = path to lvl 5 mod z gctx
  library(tidyverse)
  library(cmapR)
  library(rhdf5)
  library(CARNIVAL)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  library(viper)
  load(file = system.file("BEST_viperRegulon.rdata",package="CARNIVAL")) # loading the viper regulons
  ### load the profiles expression
  cmap_data <- get_cmap_signatures(cmap_path_to_gctx = cmap_path,sig_ids = sigs,landmark = TRUE,landmark_df = landmark)
  ### calculate tf enrichment
  TF_cmap <-runDoRothEA(cmap_data, regulon=viper_regulon, confidence_level=c('A','B','C')) # Estimating TF activities
  return(TF_cmap)
}
