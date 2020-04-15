go_path_analysis <- function(sig_ids, cmap_path_to_gctx, landmark_df, goterms) {
  ### this function calculates the NES and p.adj of the given signature ids
  ### GOterms in a list form are used
  library(tidyverse)
  library(fgsea)
  library(gage)
  library(EGSEAdata)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  
  ### first thing is to load the profiles from the GSE file
  
  profiles <- get_cmap_signatures(cmap_path_to_gctx = cmap_path_to_gctx,sig_ids = sig_ids,landmark = T,landmark_df = landmark_df)
  print("profiles loaded")
  ### change to entrez ids
  rownames(profiles) <- landmark$`Entrez ID`
  rownames(profiles) <- as.character(rownames(profiles))
  
  print("running fgsea")
  
  go_list <- apply(profiles,MARGIN = 2,fgsea,pathways = goterms,
                          minSize=10,
                          maxSize=500,
                          nperm=1000)
  
  print("fgsea finished")
  
  ### get the NES and p.adj
  
  print("preparing output")
  NES <- go_list[[1]]$NES
  padj <- go_list[[1]]$padj
  
  for (i in 2:length(go_list)) {
    
    NES <- cbind(NES,go_list[[i]]$NES)
    padj <- cbind(padj,go_list[[i]]$padj)
  }
  
  colnames(NES) <- names(go_list)
  rownames(NES) <- go_list[[1]]$pathway
  colnames(padj) <- names(go_list)
  rownames(padj) <- go_list[[1]]$pathway
  
  comb <- list(NES,padj)
  
  return(comb)
  
}
