distance_scores <- function(num_table, threshold_count, names) {
  library(GeneExpressionSignature)
  library(tidyverse)
  
  ### rank the table
  table_ranked <- apply(X = -num_table, MARGIN = 2, FUN = rank, ties.method = "random")
  
  ### create the phenodata
  pheno2 <- as.data.frame(colnames(num_table))
  rownames(pheno2) <- colnames(num_table)
  pheno_new <- new("AnnotatedDataFrame",data=pheno2)
  ### create expression set
  expr_set <- new("ExpressionSet",exprs = table_ranked, phenoData=pheno_new)
  ### calculate distances
  distances <- ScoreGSEA(expr_set , threshold_count,"avg")
  colnames(distances) <- names
  rownames(distances) <- names
  return(distances)
}
