library(AnnotationDbi)
library(tidyverse)

#Load ECFP4 similarities of train set's pairs
similarities_path_ecfp4 <- c("C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/out_vcap_train.csv",
                             "C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/out_mcf7_train.csv",
                             "C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/out_a375_train.csv",
                             "C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/out_pc3_train.csv")
#Load training sets
train_path <- c("C:/Users/user/Documents/deepSIBA/learning/data/vcap/train_test_split/train.csv",
                "C:/Users/user/Documents/deepSIBA/learning/data/mcf7/train_test_split/train.csv",
                "C:/Users/user/Documents/deepSIBA/learning/data/a375/train_test_split/train.csv",
                "C:/Users/user/Documents/deepSIBA/learning/data/pc3/train_test_split/train.csv")

#Load unique train smiles
uniq_train <- c("C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/vcap_uniq_train_smis.csv",
                "C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/mcf7_uniq_train_smis.csv",
                "C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/a375_uniq_train_smis.csv",
                "C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_5/pc3_uniq_train_smis.csv")


cell <- c('vcap','mcf7','a375','pc3')

#Thesholds to define similars in ECFP4 level and Biological distance level
ecfp4_thresh <- 0.3
bio_thresh <- 0.2

#Define output folder pattern
out <- "C:/Users/user/Documents/deepSIBA/article_results/article_data/"

for (i in 1:length(cell)){
  similarity <- as.matrix(read.csv(similarities_path_ecfp4[i]) %>% select(-X))
  names <- read.csv(uniq_train[i])
  colnames(similarity) <- names$smiles
  rownames(similarity) <- names$smiles
  similarity <- 1-similarity
  train <- read.csv(train_path[i])
  dist <- NULL
  for (k in 1:nrow(train)){
    ind1 <- which(rownames(similarity)==as.character(train$rdkit.x[k]))
    ind2 <- which(colnames(similarity)==as.character(train$rdkit.y[k]))
    if (is_empty(ind1) & is_empty(ind2)){
      ind1 <- which(rownames(similarity)==as.character(train$rdkit.y[k]))
      ind2 <- which(colnames(similarity)==as.character(train$rdkit.x[k]))
    }
    dist[k] <- similarity[ind1,ind2]
  }
  train$dist <- NULL
  train$dist <- dist
  dist <- NULL
  gc()
  
  train <- train %>% filter(dist>0)
  png(file=paste0(out,"_",cell[i],"/",cell[i],"_ecfp4_vs_go_train.png"),width=8,height=6,units = "in",res=300)
  plot(train$value/2,train$dist,pch = '.', cex = 1.3, xlab = "GO term distances", ylab = "ECFP4 distances",
       main = toupper(cell[i]),ylim=c(0,1),xlim=c(0,1))
  abline(h=ecfp4_thresh, col="red")
  abline(v=bio_thresh, col="red")
  dev.off()
}