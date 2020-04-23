#comp= smiles
#initial=data frame with sinatures
#pa_nes enrichmet scores for pathways
vectorized_borda <- function(comp,initial,pa_nes){
  sig <- initial$sig_id[which(initial$RDkit_smiles %in% comp)]
  ind <- apply(as.matrix(sig),MARGIN=1,FUN = grep,colnames(pa_nes))
  x <- as.matrix(pa_nes[,ind])
  x <- apply(x,MARGIN=2,rank,ties.method ="random")
  pheno <- as.matrix(rep(comp,ncol(x)))
  rownames(pheno) <- colnames(x)
  pheno <- new("AnnotatedDataFrame",data=as.data.frame(pheno))
  exprset <- new("ExpressionSet",exprs=x,phenoData=pheno)
  ### merge
  
  merged <- RankMerging(exprset, MergingDistance = "Spearman", weighted = TRUE)
  
  ### result
  merged_result <- exprs(merged)
  return(merged_result)
}
test2 <- future_apply(as.matrix(smis),MARGIN = 1,FUN = vectorized_borda,initial,pa_nes)
rownames(test2) <- rownames(pa_nes)
colnames(test2) <- smis
saveRDS(test2,"Documents/aggregated_parank_cmap_down_in_top_q1.rds")