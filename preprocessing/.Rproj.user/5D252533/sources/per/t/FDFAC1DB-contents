distance_scores_all <- function(num_table1,num_table2, threshold_count, names1,names2) {
  library(GeneExpressionSignature)
  library(tidyverse)
  
  ### rank the table
  table_ranked1 <- apply(X = -num_table1, MARGIN = 2, FUN = rank, ties.method ="random")
  table_ranked2 <- apply(X = -num_table2, MARGIN = 2, FUN = rank, ties.method ="random")
  final_dist <- matrix(nrow=NCOL(num_table1),ncol=NCOL(num_table2))
  for (i in 1:NCOL(num_table1)) {
    for (j in 1:NCOL(num_table2)) {
      ### create expression set
      merged_table <- cbind(table_ranked1[,i],table_ranked2[,j])
      colnames(merged_table) <- c(colnames(table_ranked1)[i],colnames(table_ranked2)[j])
      pheno <- as.data.frame(colnames(merged_table))
      rownames(pheno) <- colnames(merged_table)
      pheno_new <- new("AnnotatedDataFrame",data=pheno)
      expr_set <- new("ExpressionSet",exprs = merged_table, phenoData=pheno_new)
      ### calculate distances
      distances <- ScoreGSEA(expr_set , threshold_count,"avg")
      final_dist[i,j] <-distances[1,2] #or (2,1) just not the primary diagonal 
    }
    print((i/ncol(num_table1))*100)
  }
  #colnames(final_dist) <- names1
  #rownames(final_dist) <- names2
  return(final_dist)
}
