# for testing
library(tidyverse)

get_com_sigs_2 <- function(gos,pa,tb_go=10,tb_pa=10, sig_ids) {
  
  #library(tidyverse)
  #library(rhdf5)
  #library(org.Hs.eg.db)
  
  # load the GO library
  #library(GO.db)
  
  # extract a named vector of all terms
  #goterms <- Term(GOTERM)
  #goterms <- as.data.frame(goterms) %>% rownames_to_column("id")
  go <- gos[,sig_ids] 
  go <- apply(go,MARGIN = 2,FUN = rank,ties.method = "random")
  
  majority <- function(x,nrow,tb){
    # work for bot
    bot <- sum((x<=tb + 0))/length(x)
    top <- sum((x>=(nrow-tb+1)))/length(x)
    return(cbind(bot,top))
  }
  
  go_tb <- t(apply(go,MARGIN = 1,FUN = majority,nrow = nrow(go),tb = tb_go))
  colnames(go_tb) <- c("bot","top")
  
  
  go_bot <- go_tb[,1]
  go_top <- go_tb[,2]
  
  pathways <- pa[,sig_ids]
  pathways <- apply(pathways,MARGIN = 2,FUN = rank,ties.method = "random")
  pa_tb <- t(apply(pathways,MARGIN = 1,FUN = majority,nrow = nrow(pathways),tb = tb_pa))
  colnames(pa_tb) <- c("bot","top")
  
  pa_bot <- pa_tb[,1]
  pa_top <- pa_tb[,2]
  
  #Return output list
  output <- list(go_bot,go_top,pa_bot,pa_top)
  names(output) <- c("BotGos","TopGos","BotPAs","TopPAs")
  return(output)
}

#need to add sig+rdkit dataframe (mcf7) and goterms name to id mapping
# read mcf7 signatures
mcf7 <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/mcf7_all_sigs.rds")
# read pert to rdkit
pert <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/utility/pert_id_to_rdkit.rds")
mcf7 <- left_join(mcf7,pert)
mcf7 <- mcf7 %>% filter(!is.na(rdkit))

# read go term scores and pathway scores
path_pa <- "C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/nes_compounds_all.rds"
path_gos <- "C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/all_compounds_go.rds"
pathway_scores <- readRDS(path_pa)
go_scores <- readRDS(path_gos)
query_val <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/test_mcf7.csv")
thresh <- 0.2
query_val <- query_val %>% filter(preds <= thresh) 
# for each drug in the query count up the similars
query_val <- query_val %>% group_by(x2) %>% mutate(counts = n_distinct(x1)) %>% ungroup()
# set minimum number of similars for go term overlap
no_overlap <- 5
query_val <- query_val %>% filter(counts >= no_overlap) 

# what query drugs are left
val_drugs_left <- unique(as.character(query_val$x2))
length(val_drugs_left)
neighbors <- c(5,8,10,12,15,20,25,30,35,40,50,60,70,85,100,120,150,175,200,225)
# top bot gos and pas
tb_go <- 170
tb_pa <- 10
results_train <- NULL
results <- NULL
dist_pb <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_pb.rds")
dist_pt <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_pt.rds")
dist_gb <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_gb.rds")
dist_gt <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_gt.rds")
for (i in 1:length(val_drugs_left)){
  # keep in filt the similars of each query in the loop
  filt <- query_val[which(query_val$x2 == val_drugs_left[i]),]
  # add rdkit and sig ids in filt
  filt <- left_join(filt,mcf7, by = c("x1"="rdkit"))
  
  
  
  diff <- abs(nrow(filt) - neighbors)
  pos <- which(diff==min(diff))[1]
  # now get the common pas and gos for the similars of the query
  train_overlap <- get_com_sigs_2(gos = go_scores,
                                  pa = pathway_scores,sig_ids = filt$sig_id,
                                  tb_go = tb_go, tb_pa = tb_pa)
  
  gb_p <- NULL
  gt_p <- NULL
  for (k in 1:length(train_overlap$BotGos)) {
    gb_p[k] <- length(which(dist_gb[[pos]][k,]>=train_overlap$BotGos[k]))/5000
    gt_p[k] <- length(which(dist_gt[[pos]][k,]>=train_overlap$TopGos[k]))/5000
  }
  train_overlap$BotGos <- cbind(train_overlap$BotGos,gb_p)
  train_overlap$TopGos <- cbind(train_overlap$TopGos,gt_p)
  
  pb_p <- NULL
  pt_p <- NULL
  for (k in 1:length(train_overlap$BotPAs)) {
    pb_p[k] <- length(which(dist_pb[[pos]][k,]>=train_overlap$BotPAs[k]))/5000
    pt_p[k] <- length(which(dist_pt[[pos]][k,]>=train_overlap$TopPAs[k]))/5000
  }
  
  train_overlap$BotPAs <- cbind(train_overlap$BotPAs,pb_p)
  train_overlap$TopPAs <- cbind(train_overlap$TopPAs,pt_p)
  results_train[[i]] <- train_overlap
  bot_gos <- names(which(results_train[[i]]$BotGos[,1] >= 0.65 & results_train[[i]]$BotGos[,2] <= 0.01))
  top_gos <- names(which(results_train[[i]]$TopGos[,1] >= 0.65 & results_train[[i]]$TopGos[,2] <= 0.01))
  bot_pas <- names(which(results_train[[i]]$BotPAs[,1] >= 0.65 & results_train[[i]]$BotPAs[,2] <= 0.01))
  top_pas <- names(which(results_train[[i]]$TopPAs[,1] >= 0.65 & results_train[[i]]$TopPAs[,2] <= 0.01))
  print(bot_pas)
  
  results[[i]] <- list(bot_gos,top_gos,bot_pas,top_pas)
  
}
