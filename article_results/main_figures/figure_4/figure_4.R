library(tidyverse)
library(viridis)
library(reshape2)

# set path to cmap gctx file
ds_path <- "C:/Users/user/Documents/phd/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
landmark <- read_tsv(file = "C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/cmap/cmap_landmark_genes.txt")

### read the kd consensus signatures

kd <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/figure_extra_data/figure_4/tf_kd_cgs.rds")

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
### keep the myc_kd

myc_kd <- kd %>% filter(pert_iname == "MYC")

### load the profiles

profiles_myc <- get_cmap_signatures(cmap_path_to_gctx = ds_path, sig_ids = myc_kd$sig_id,landmark = T,landmark_df = landmark)

### calculate pathway scores

pathways_myc <- kegg_path_analysis(sig_ids = myc_kd$sig_id,cmap_path_to_gctx = ds_path,landmark_df = landmark)

pathways_myc <- pathways_myc[[1]]

### calculate go terms

gos <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/utility/goterm_annotation.rds")
go_myc <- go_path_analysis(sig_ids = myc_kd$sig_id,cmap_path_to_gctx = ds_path,landmark_df = landmark,goterms = gos)

go_myc <- go_myc[[1]]


### calculate distances with scoregsea

genes_dist <- distance_scores(num_table = profiles_myc,threshold_count = 25, names= myc_kd$cell_id)

paths_dist <- distance_scores(num_table = pathways_myc,threshold_count = 10, names = myc_kd$cell_id)

go_dist <- distance_scores(num_table = go_myc, threshold_count = 25, names = myc_kd$cell_id)

### calculate correlations

colnames(profiles_myc) <- myc_kd$cell_id
colnames(pathways_myc) <- myc_kd$cell_id
colnames(go_myc) <- myc_kd$cell_id

genes_cor <- round(cor(profiles_myc,method = "spearman"),2)

pathways_cor <- round(cor(pathways_myc,method = "spearman"),2)

go_cor <- round(cor(go_myc,method = "spearman"),2)

### plot correlations

melted_genes_cor <- melt(genes_cor)

ggplot(data = melted_genes_cor, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()+scale_fill_viridis(limits = c(0,1)) + labs(title = "gene-level spearman")

melted_pathways_cor <- melt(pathways_cor)

ggplot(data = melted_pathways_cor, aes(x=Var1, y=Var2, fill= value)) + 
  geom_tile()+scale_fill_viridis(limits = c(0,1)) + labs(title = "KEGG pathway level spearman")

melted_go_cor <- melt(go_cor)

ggplot(data = melted_go_cor, aes(x=Var1, y=Var2, fill= value)) + 
  geom_tile()+scale_fill_viridis(limits = c(0,1)) + labs(title = "GO BP level spearman")


### plot distances

melted_genes_dist <- melt(genes_dist)
png(file="result_fig2a.png",width=6.5,height=6,units = "in",res=300)
ggplot(data = melted_genes_dist, aes(x=Var1, y=Var2, fill= value/2)) + 
  geom_tile(show.legend = F)+scale_fill_viridis(limits = c(0,1),direction = -1) + labs( fill = "")+xlab("Cell line")+
  ylab("Cell line") + ggtitle("A") +theme(plot.title = element_text(hjust=0))
dev.off()
melted_paths_dist <- melt(paths_dist)

ggplot(data = melted_paths_dist, aes(x=Var1, y=Var2, fill= 1-value)) + 
  geom_tile()+scale_fill_viridis(limits = c(0,1)) + labs(title = "KEGG Pathway Level GSEA scores")

melted_go_dist <- melt(go_dist)
png(file="result_fig2b.png",width=7.5,height=6,units = "in",res=300)
ggplot(data = melted_go_dist, aes(x=Var1, y=Var2, fill= value/2)) + 
  geom_tile()+scale_fill_viridis(limits = c(0,1),direction = -1) + 
  labs( fill = "Distance") + xlab("Cell line")+ylab("Cell line")+
  ggtitle("B") +theme(plot.title = element_text(hjust=0))

dev.off()

