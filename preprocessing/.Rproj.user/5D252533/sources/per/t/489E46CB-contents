kegg_path_analysis <- function(sig_ids, cmap_path_to_gctx, landmark_df) {
  ### this function calculates the NES and p.adj of the given signature ids
  ### 125 KEGG pathways are used
  library(tidyverse)
  library(fgsea)
  library(gage)
  library(EGSEAdata)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  
  ### first thing is to load the profiles from the GSE file
  
  profiles <- get_cmap_signatures(cmap_path_to_gctx = cmap_path_to_gctx,sig_ids = sig_ids,landmark = T,landmark_df = landmark_df)
  print("profiles loaded")
  ### load the gene sets
  print("loading gene sets")
  egsea.data(species = "human",returnInfo = TRUE)
  rownames(profiles) <- landmark$`Entrez ID`
  rownames(profiles) <- as.character(rownames(profiles))
  
  print("running fgsea")
  ### run the analysis
  pathway_list <- apply(profiles,MARGIN = 2,fgsea,pathways = kegg.pathways$human$kg.sets,
                           minSize=10,
                           maxSize=500,
                           nperm=1000)
  print("fgsea finished")
  ### get the NES and p.adj
  
  print("preparing output")
  NES <- pathway_list[[1]]$NES
  padj <- pathway_list[[1]]$padj
  
  for (i in 2:length(pathway_list)) {
    
    NES <- cbind(NES,pathway_list[[i]]$NES)
    padj <- cbind(padj,pathway_list[[i]]$padj)
  }
  
  colnames(NES) <- names(pathway_list)
  rownames(NES) <- pathway_list[[1]]$pathway
  colnames(padj) <- names(pathway_list)
  rownames(padj) <- pathway_list[[1]]$pathway
  
  comb <- list(NES,padj)
  
  return(comb)
}

