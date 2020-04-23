library(tidyverse)
library(doFuture)
library(cancerTiming)

registerDoFuture()
plan(multiprocess,workers = 12)

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


get_cmap_signatures <- function(cmap_path_to_gctx, sig_ids, landmark = TRUE, landmark_df = NULL) {
  
  
  library(tidyverse)
  library(cmapR)
  library(rhdf5)
  library(AnnotationDbi)
  library(org.Hs.eg.db)
  
  ds_path <- cmap_path_to_gctx
  if (landmark == TRUE) {
    
    cmap_gctx <- parse.gctx(ds_path,rid = as.character(landmark_df$`Entrez ID`), cid = sig_ids)
    cmap <- cmap_gctx@mat
    
    cmap <- cmap[as.character(landmark_df$`Entrez ID`),]
    
    rownames(cmap) <- landmark_df$Symbol
  }
  
  if (landmark == FALSE) {
    
    cmap_gctx <- parse.gctx(ds_path, cid = sig_ids)
    cmap <- cmap_gctx@mat
    
    entrez <- rownames(cmap)
    anno <- AnnotationDbi::select(org.Hs.eg.db,
                                  keys = entrez,
                                  columns = c("SYMBOL", "GENENAME","ENTREZID"),
                                  keytype = "ENTREZID")
    
    anno <- anno %>%
      filter(!is.na(SYMBOL))
    
    cmap <- cmap[anno$ENTREZID,]
    
    rownames(cmap) <- anno$SYMBOL
  }
  
  
  return(cmap)
  
}
all <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/processed_data/cmap_with_RDkits_initial.rds")
go <- readRDS("article_data/all_compounds_go.rds")
thresh <- c(10,20,30,40,50)
# instructions to download this file are found at the preprocessing readme
ds_path <- "C:/Users/user/Documents/phd/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
landmark <- read_tsv(file = "C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/cmap/cmap_landmark_genes.txt")
#### MCF7

cell_line <- "MCF7"

calculate_figures <- function(cell_line,all,go,thres,ds_path,landmark){
  df_mcf7 <- all %>% filter(cell_id == cell_line) %>% 
    filter(quality == "1")
  
  go_mcf7 <- go[,df_mcf7$sig_id]
  
  dist_mcf7_go <- foreach(thres = thresh) %dopar% {
    distance_scores(num_table = go_mcf7,threshold_count = thres,names = colnames(go_mcf7))
  }
  for (i in 1:length(dist_mcf7_go)) {
    dist_mcf7_go[[i]] <- dist_mcf7_go[[i]]/2
  }
  
  ld <- length(dist_mcf7_go)
  dimd <- dim(dist_mcf7_go[[1]])
  distance <- do.call(cbind,dist_mcf7_go)
  distance <- array(distance,c(dim=dimd,ld))
  
  sd_dist_mcf7_go <- apply(distance, c(1,2), sd, na.rm = TRUE)
  
  #hist(sd_dist_mcf7_go,col = "gray",main = 'histogram of standard deviation for quality 1 pairs in MCF7',xlab = "standard deviation")
  
  mean_dist_mcf7_go <- apply(distance, c(1,2), mean, na.rm = TRUE)
  
  mean_dist_mcf7_go[lower.tri(mean_dist_mcf7_go,diag = T)] <- 666
  
  colnames(mean_dist_mcf7_go) <- as.character(df_mcf7$pert_id)
  rownames(mean_dist_mcf7_go) <- as.character(df_mcf7$pert_id)
  
  mean_dist_mcf7_go <- reshape2::melt(mean_dist_mcf7_go)
  
  mean_dist_mcf7_go <- mean_dist_mcf7_go %>% filter(value != 666)
  
  gene_mcf7 <- get_cmap_signatures(cmap_path_to_gctx = ds_path,sig_ids = df_mcf7$sig_id,landmark = T,landmark_df = landmark)
  
  dist_mcf7_gene <- foreach(thres = thresh) %dopar% {
    distance_scores(num_table = gene_mcf7,threshold_count = thres,names = colnames(gene_mcf7))
  }
  
  for (i in 1:length(dist_mcf7_gene)) {
    dist_mcf7_gene[[i]] <- dist_mcf7_gene[[i]]/2
  }
  
  ld <- length(dist_mcf7_gene)
  dimd <- dim(dist_mcf7_gene[[1]])
  distance <- do.call(cbind,dist_mcf7_gene)
  distance <- array(distance,c(dim=dimd,ld))
  mean_dist_mcf7_gene <- apply(distance, c(1,2), mean, na.rm = TRUE)
  
  
  mean_dist_mcf7_gene[lower.tri(mean_dist_mcf7_gene,diag = T)] <- 666
  
  colnames(mean_dist_mcf7_gene) <- as.character(df_mcf7$pert_id)
  rownames(mean_dist_mcf7_gene) <- as.character(df_mcf7$pert_id)
  
  mean_dist_mcf7_gene <- reshape2::melt(mean_dist_mcf7_gene)
  mean_dist_mcf7_gene$value <- mean_dist_mcf7_gene$value/2
  #saveRDS(mean_dist_mcf7_gene,"gene_distances.rds")
  mean_dist_mcf7_gene <- mean_dist_mcf7_gene %>% filter(value != 666)
  return(list(mean_dist_mcf7_go,mean_dist_mcf7_gene,sd_dist_mcf7_go))
}
mcf7 <- calculate_figures(cell_line = "MCF7",all = all,go = go,thres = thresh,ds_path = ds_path,landmark = landmark)
mcf7_go <- mcf7[[1]]
mcf7_gene <- mcf7[[2]]
mcf7_gene <- mcf7_gene %>% filter(value != 333)
mcf7_sd <- mcf7[[3]]

pc3 <- calculate_figures(cell_line = "PC3",all = all,go = go,thres = thresh,ds_path = ds_path,landmark = landmark)
pc3_go <- pc3[[1]]
pc3_gene <- pc3[[2]]
pc3_gene <- pc3_gene %>% filter(value != 333)
pc3_sd <- pc3[[3]]

a375 <- calculate_figures(cell_line = "A375",all = all,go = go,thres = thresh,ds_path = ds_path,landmark = landmark)
a375_go <- a375[[1]]
a375_gene <- a375[[2]]
a375_gene <- a375_gene %>% filter(value != 333)
a375_sd <- a375[[3]]

vcap <- calculate_figures(cell_line = "VCAP",all = all,go = go,thres = thresh,ds_path = ds_path,landmark = landmark)
vcap_go <- vcap[[1]]
vcap_gene <- vcap[[2]]
vcap_gene <- vcap_gene %>% filter(value != 333)
vcap_sd <- vcap[[3]]

png(file="supl_fig_2_mcf7.png",width=8,height=6,units = "in",res=300)
hist(mcf7_sd,col = "gray",xlab = "Standard deviation",breaks = seq(0,0.18,0.01),main = NULL,ylim = c(0,200000))
title("MCF7",adj = 0)
dev.off()

png(file="supl_fig_2_pc3.png",width=8,height=6,units = "in",res=300)
hist(pc3_sd,col = "gray",xlab = "Standard deviation",main = NULL,breaks = seq(0,0.18,0.01),ylim = c(0,180000))
title("PC3",adj = 0)
dev.off()

png(file="supl_fig_2_a375.png",width=8,height=6,units = "in",res=300)
hist(a375_sd,col = "gray",xlab = "standard deviation",main = NULL,breaks = seq(0,0.18,0.01),ylim = c(0,180000))
title("A375",adj = 0)
dev.off()

png(file="supl_fig_2_vcap.png",width=8,height=6,units = "in",res=300)
hist(vcap_sd,col = "gray",xlab = "standard deviation",main = NULL,breaks = seq(0,0.18,0.01), ylim = c(0,300000))
title("VCAP",adj = 0)
dev.off()

png(file="supl_fig_3_mcf7.png",width=8,height=6,units = "in",res=300)
plot(mcf7_gene$value*2,mcf7_go$value,pch = '.', cex = 1, xlab = "Gene distance", ylab = "GO term distance",xlim = c(0,1),ylim = c(0,1),
     main = NULL, xaxs = "i", yaxs = "i")
title("MCF7",adj = 0)
legend("topleft", 
       legend = "Pearson's r = 0.815", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()

png(file="supl_fig_3_pc3.png",width=8,height=6,units = "in",res=300)
plot(pc3_gene$value*2,pc3_go$value,pch = '.', cex = 1, xlab = "Gene distance", ylab = "GO term distance",xlim = c(0,1),ylim = c(0,1),
     main = NULL, xaxs = "i", yaxs = "i")
title("PC3",adj = 0)
legend("topleft", 
       legend = "Pearson's r = 0.797", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()


png(file="supl_fig_3_a375.png",width=8,height=6,units = "in",res=300)
plot(a375_gene$value*2,a375_go$value,pch = '.', cex = 1, xlab = "Gene distance", ylab = "GO term distance",xlim = c(0,1),ylim = c(0,1),
     main = NULL, xaxs = "i", yaxs = "i")
title("A375",adj = 0)
legend("topleft", 
       legend = "Pearson's r = 0.760", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()


png(file="supl_fig_3_vcap.png",width=8,height=6,units = "in",res=300)
plot(vcap_gene$value*2,vcap_go$value,pch = '.', cex = 1, xlab = "Gene distance", ylab = "GO term distance",xlim = c(0,1),ylim = c(0,1),
     main = NULL, xaxs = "i", yaxs = "i")
title("VCAP",adj = 0)
legend("topleft", 
       legend = "Pearson's r = 0.904", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()