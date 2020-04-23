library(tidyverse)
library(GO.db)
path_pa <- "C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/nes_compounds_all.rds"
path_gos <- "C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/all_compounds_go.rds"
pathway_scores <- readRDS(path_pa)
go_scores <- readRDS(path_gos)

dist_pb <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_pb.rds")
dist_pt <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_pt.rds")
dist_gb <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_gb.rds")
dist_gt <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/distributions_gt.rds")
get_com_sigs_2 <- function(gos,pa,tb_go=10,tb_pa=10, sig_ids, majority_thresh = 0.5) {
  
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

# extract a named vector of all terms
goterms <- Term(GOTERM)
goterms <- as.data.frame(goterms)
goterms <- goterms %>% rownames_to_column("id")
mcf7 <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_6/mcf7_all_sigs.rds")

# read pert to rdkit
pert <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/utility/pert_id_to_rdkit.rds")
mcf7 <- left_join(mcf7,pert)
mcf7 <- mcf7 %>% filter(!is.na(rdkit))
pa_tb <- 10
go_tb <- 170
# neighbors have to do with the p value distributions, the closest number of neighbors is selected (pos)
neighbors <- c(5,8,10,12,15,20,25,30,35,40,50,60,70,85,100,120,150,175,200,225)
thresholds <- seq(0.6,1.0,0.05)
d_th <- seq(0.2,0.2,0.01)
kn <- seq(5,5,1)
results_all <- data.frame(matrix(nrow = length(kn),ncol = 9))
colnames(results_all) <- c("overlap_bot_go_%","common_bot_go",
                           "overlap_top_go_%","common_top_go",
                           "overlap_bot_pa_%","common_bot_pa",
                           "overlap_top_pa_%","common_top_pa","threshold") 
for (n in 1:length(thresholds)) {
  
  
  query_val <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_9_10/val_mcf7.csv")
  thresh <- d_th[1]
  query_val <- query_val %>% filter(preds <= thresh) 
  # for each drug in the query count up the similars
  query_val <- query_val %>% group_by(x2) %>% mutate(counts = n_distinct(x1)) %>% ungroup()
  # set minimum number of similars for go term overlap
  no_overlap <- kn[1]
  query_val <- query_val %>% filter(counts >= no_overlap) 
  
  # what query drugs are left
  val_drugs_left <- unique(as.character(query_val$x2))
  length(val_drugs_left)
  
  # top bot gos and pas
  tb_go <- 170
  tb_pa <- 10
  results_train <- NULL
  results_query <- NULL
  results <- data.frame(matrix(nrow = length(val_drugs_left),ncol = 9))
  colnames(results) <- c("overlap_bot_go_%","common_bot_go",
                         "overlap_top_go_%","common_top_go",
                         "overlap_bot_pa_%","common_bot_pa",
                         "overlap_top_pa_%","common_top_pa","number_similars")
  
  for (i in 1:length(val_drugs_left)){
    # keep in filt the similars of each query in the loop
    filt <- query_val[which(query_val$x2 == val_drugs_left[i]),]
    # add rdkit and sig ids in filt
    filt <- left_join(filt,mcf7, by = c("x1"="rdkit"))
    # what is the sig id of the query in the loop
    sig_val <- mcf7$sig_id[which(mcf7$rdkit %in% filt$x2[1])]
    
    # get go terms of the query
    go_query <- go_scores[,sig_val]
    # turn to DF
    go_query <- as.data.frame(go_query) %>% rownames_to_column("id")
    # add names of goterms
    go_query <- left_join(go_query,goterms)
    # order
    go_query <- go_query[order(go_query$go_query,decreasing = F),]
    
    go_query_bot <- go_query[1:tb_go,]
    go_query_top <- go_query[nrow(go_query):(nrow(go_query)-tb_go+1),]
    
    # get pathways of query
    pa_query <- pathway_scores[,sig_val]
    # turn to DF order and get top and bot
    pa_query <- as.data.frame(pa_query) %>% rownames_to_column("pathways")
    pa_query <- pa_query[order(pa_query$pa_query,decreasing = F),]
    pa_query_bot <- pa_query[1:tb_pa,]
    pa_query_top <- pa_query[nrow(pa_query):(nrow(pa_query)-tb_pa+1),]
    
    print(nrow(filt))
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
    
    query <- list(go_query_bot,go_query_top,pa_query_bot,pa_query_top)
    names(query) <- c("gb","gt","pb","pt")
    results_train[[i]] <- train_overlap
    results_query[[i]] <- query
    
    bot_gos <- names(which(results_train[[i]]$BotGos[,1] >= thresholds[n] & results_train[[i]]$BotGos[,2] <= 0.01))
    top_gos <- names(which(results_train[[i]]$TopGos[,1] >= thresholds[n] & results_train[[i]]$TopGos[,2] <= 0.01))
    bot_pas <- names(which(results_train[[i]]$BotPAs[,1] >= thresholds[n] & results_train[[i]]$BotPAs[,2] <= 0.01))
    top_pas <- names(which(results_train[[i]]$TopPAs[,1] >= thresholds[n] & results_train[[i]]$TopPAs[,2] <= 0.01))
    
    # number of common bot and top gos
    no_common_bot_gos <- length(bot_gos)
    no_common_top_gos <- length(top_gos)
    # number of common bot and top pathways
    no_common_bot_pas <- length(bot_pas)
    no_common_top_pas <- length(top_pas)
    
    # how many of the common pas and gos at top and bot are in the queries top 10
    
    overlap_bot_go <- length(which(bot_gos %in% results_query[[i]]$gb$id))/no_common_bot_gos
    overlap_top_go <- length(which(top_gos %in% results_query[[i]]$gt$id))/no_common_top_gos
    overlap_bot_pa <- length(which(bot_pas %in% results_query[[i]]$pb$pathways))/no_common_bot_pas
    overlap_top_pa <- length(which(top_pas %in% results_query[[i]]$pt$pathways))/no_common_top_pas
    print(bot_pas)
    results[i,] <- c(overlap_bot_go,no_common_bot_gos,overlap_top_go,no_common_top_gos,
                     overlap_bot_pa,no_common_bot_pas,overlap_top_pa,no_common_top_pas,nrow(filt))
    
    
    
  }
  results_all[n,c(1,3,5,7)] <- apply(results[,c(1,3,5,7)],MARGIN = 2, mean, na.rm=T)
  results_all[n,c(2,4,6,8)] <- apply(results[,c(2,4,6,8)],MARGIN = 2, median, na.rm=T)
  results_all[n,9] <- thresholds[n]
  print(k)
  
}

png(file="supl_fig_9_down_prec.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$`overlap_bot_pa_%`,"o",col = "black",xlim = c(0.178,0.45),lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",
     ylim = c(-0.01,1),xlab = "Distance threshold",ylab="Average precision")
title("A",adj = 0)
legend("topright", 
       legend = "f_th = 0.65, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()

png(file="supl_fig_9_up_prec.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$`overlap_top_pa_%`,"o",col = "black",xlim = c(0.178,0.45),lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",
     ylim = c(-0.01,1),xlab = "Distance threshold",ylab="Average precision")
title("B",adj = 0)
legend("topright", 
       legend = "f_th = 0.6, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()

png(file="supl_fig_9_down_numbers.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$common_bot_pa,"o",col = "black",xlab = "Distance threshold",ylab="Average length of inferred signature",
     lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",ylim = c(-0.1,5.1),xlim = c(0.179,0.451))
title("C",adj = 0)
legend("topright", 
       legend = "f_th=0.6, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()



png(file="supl_fig_9_up_numbers.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$common_top_pa,"o",col = "black",xlab = "Distance threshold",ylab="Average length of inferred signature",
     lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",ylim = c(-0.1,5.1),xlim = c(0.179,0.451))
title("D",adj = 0)
legend("topright", 
       legend = "f_th = 0.6, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()


png(file="supl_fig_10_down_prec.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$`overlap_bot_pa_%`,"o",col = "black",xlim = c(0.59,1.01),lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",
     ylim = c(-0.01,1.01),xlab = "Frequency threshold",ylab="Average precision")
title("A",adj = 0)
legend("topleft", 
       legend = "d_th = 0.2, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()

png(file="supl_fig_10_up_prec.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$`overlap_top_pa_%`,"o",col = "black",xlim = c(0.59,1.01),lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",
     ylim = c(-0.01,1.01),xlab = "Frequency threshold",ylab="Average precision")
title("B",adj = 0)
legend("topleft", 
       legend = "d_th = 0.2, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()

png(file="supl_fig_10_down_numbers.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$common_bot_pa,"o",col = "black",xlab = "Frequency threshold",ylab="Average length of inferred signature",
     lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",ylim = c(-0.1,5.1),xlim = c(0.59,1.01))
title("C",adj = 0)
legend("topright", 
       legend = "d_th=0.2, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()



png(file="supl_fig_10_up_numbers.png",width=8,height=6,units = "in",res=300)
plot(results_all$threshold,results_all$common_top_pa,"o",col = "black",xlab = "Frequency threshold",ylab="Average length of inferred signature",
     lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i",ylim = c(-0.1,5.1),xlim = c(0.59,1.01))
title("D",adj = 0)
legend("topright", 
       legend = "d_th = 0.2, k = 5, p_th = 0.01", 
       col = "black",
       pch = '',
       bty = "o", 
       pt.cex = 2, 
       cex = 1, 
       text.col = "black", 
       horiz = T )
dev.off()
