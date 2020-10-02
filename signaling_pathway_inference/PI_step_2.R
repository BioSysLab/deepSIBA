library(tidyverse)

# extract a named vector of all terms
# read all signature data frame
# the file below can be found at deepSIBA drive, data preprocessing
all <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/processed_data/initial_signatures_with_mycranks.rds")
#read pert id to rdkit mapping
pert <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/utility/pert_id_to_rdkit.rds")
all <- left_join(all,pert)
all <- all %>% filter(!is.na(rdkit))

# read pathway scores
path_pa <- "pathway_inference_data/nes_compounds_all.rds"
# path to the neighbor file from step 1
neighbor_file <- "C:/Users/user/Documents/deepSIBA/results/inference_test_jan_0/results_trainingset.csv"

# read precalculated p value distributions for up and downregulated pathways
dist_pb <- readRDS("pathway_inference_data/distributions_pb.rds")
dist_pt <- readRDS("pathway_inference_data/distributions_pt.rds")

get_com_sigs_2 <- function(pa,tb_pa=10, sig_ids, majority_thresh = 0.5) {
  
  #library(tidyverse)
  #library(rhdf5)
  #library(org.Hs.eg.db)
  
  # load the GO library
  #library(GO.db)
  
  # extract a named vector of all terms
  #goterms <- Term(GOTERM)
  #goterms <- as.data.frame(goterms) %>% rownames_to_column("id")
  
  majority <- function(x,nrow,tb){
    # work for bot
    bot <- sum((x<=tb + 0))/length(x)
    top <- sum((x>=(nrow-tb+1)))/length(x)
    return(cbind(bot,top))
  }
  
  
  
  pathways <- pa[,sig_ids]
  pathways <- apply(pathways,MARGIN = 2,FUN = rank,ties.method = "random")
  pa_tb <- t(apply(pathways,MARGIN = 1,FUN = majority,nrow = nrow(pathways),tb = tb_pa))
  colnames(pa_tb) <- c("bot","top")
  
  pa_bot <- pa_tb[,1]
  pa_top <- pa_tb[,2]
  
  #Return output list
  output <- list(pa_bot,pa_top)
  names(output) <- c("BotPAs","TopPAs")
  return(output)
}
infer_pathways <- function(cell_line, path_pa, neighbor_file, all, n_min_neighbors, tb_pa, dist_pb, dist_pt,
                           fr_thresh, p_val_thresh, n_trials){
  # cell line is the cell line of choice
  # path_pa is the path to the file with all the pathway scores
  # neighbor_file is the path to the neighbor file produced from PI_step_1.ipynb
  # all is the data file with the unique sig_ids
  # n_min_neighbors are the minimum number of neighbors required to perform the inference (rec_value = 5)
  # tb_pa the max length of the top and bottom pathway list that the inference looks at (rec_value = 10)
  # dist_pb, dist_pt precomputed distributions of random frequency scores for each pathway for n=5000 runs
  # fr_thresh the frequency threshold in order to infer a pathway from the neighbors' lists (rec_value = 0.65)
  # p_val_thresh significance threshold (rec_value = 0.01)
  # n_trials = 5000 for the precomputed frequency score distributions
  # read pathway scores
  pathway_scores <- readRDS(path_pa)
  # read neighbor file
  neighbors <- read.csv(neighbor_file)
  # for each drug in the query count up the similars
  neighbors <- neighbors %>% group_by(query) %>% mutate(counts = n_distinct(x)) %>% ungroup()
  # set minimum number of similars for go term overlap
  neighbors <- neighbors %>% filter(counts >= n_min_neighbors) 
  # what query drugs are left
  query_drugs <- unique(as.character(neighbors$query))
  length(query_drugs)
  # set number of neighbors for p value distributions
  neighbors_dist <- c(5,8,10,12,15,20,25,30,35,40,50,60,70,85,100,120,150,175,200,225)
  results_train <- NULL
  results <- NULL
  for (i in 1:length(query_drugs)){
    i <- 1
    # keep in filt the similars of each query in the loop
    filt <- neighbors[which(neighbors$query == query_drugs[i]),]
    # add rdkit and sig ids in filt
    cell <- all %>% filter(cell_id == cell_line)
    filt <- left_join(filt,cell, by = c("x"="rdkit"))
    
    diff <- abs(nrow(filt) - neighbors_dist)
    pos <- which(diff==min(diff))[1]
    # now get the common pas and gos for the similars of the query
    train_overlap <- get_com_sigs_2(pa = pathway_scores,sig_ids = filt$sig_id,
                                     tb_pa = tb_pa)
    
    pb_p <- NULL
    pt_p <- NULL
    for (k in 1:length(train_overlap$BotPAs)) {
      pb_p[k] <- length(which(dist_pb[[pos]][k,]>=train_overlap$BotPAs[k]))/n_trials
      pt_p[k] <- length(which(dist_pt[[pos]][k,]>=train_overlap$TopPAs[k]))/n_trials
    }
    
    train_overlap$BotPAs <- cbind(train_overlap$BotPAs,pb_p)
    train_overlap$TopPAs <- cbind(train_overlap$TopPAs,pt_p)
    results_train[[i]] <- train_overlap
    bot_pas <- names(which(results_train[[i]]$BotPAs[,1] >= fr_thresh & results_train[[i]]$BotPAs[,2] <= p_val_thresh))
    top_pas <- names(which(results_train[[i]]$TopPAs[,1] >= fr_thresh & results_train[[i]]$TopPAs[,2] <= p_val_thresh))
    
    results[[i]] <- list(bot_pas,top_pas)
    names(results[[i]]) <- c("Downregulated","Upregulated")
    
    
  }
  names(results) <- as.character(query_drugs)
  return(results)
}

test <- infer_pathways(cell_line = "A375",path_pa = path_pa,neighbor_file = neighbor_file,all = all,n_min_neighbors = 5,
                       tb_pa = 10,dist_pb = dist_pb,dist_pt = dist_pt,fr_thresh = 0.65,p_val_thresh = 0.01,n_trials = 5000)


files <- list.files("C:/Users/user/Documents/deepSIBA/results/cannabis_pathways_a375/",all.files = T,recursive = T,full.names = T)
cell <- "A375"
names <- c("CBC","CBCA","CBCV","CBCVA","CBL","CBLA","CBD","CBDM","CBDA","CBD-C1","CBD-C4","CBDV","CBDVA","CBDP","6α-OH CBD",
           "CBEA-B","CBE","CBG","CBGM","CBGA","CBGAM","CBGV","CBGVA","CBGO","CBGOA","CBND","CBN","CBNA","CBV",
           "CBT","Δ8-THC","Δ8-THCA","THC","THC-C4","THCA-A","THCA-C4","THC-C1","THCA-C1-A","THCA-C1-B","THCV","THCVA",
           "Δ9-THCP","CBF","CBR","CBT","DCBF","cis-THC")
smiles <- c("CCCCCC1=CC(=C2C=CC(OC2=C1)(C)CCC=C(C)C)O","CCCCCC1=CC2=C(C=CC(O2)(C)CCC=C(C)C)C(=C1C(=O)O)O",
           "CCCC1=CC(=C2C=CC(OC2=C1)(C)CCC=C(C)C)O","CCCC1=CC2=C(C=CC(O2)(C)CCC=C(C)C)C(=C1C(=O)O)O",
           "CCCCCC1=CC(=C2C3C4C(C3(C)C)CCC4(OC2=C1)C)O","CCCCCC1=CC2=C(C3C4C(C3(C)C)CCC4(O2)C)C(=C1C(=O)O)O",
           "CCCCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(=C)C)C)O","CCCCCC1=CC(=C(C(=C1)OC)C2C=C(CCC2C(=C)C)C)O",
           "CCCCCC1=CC(=C(C(=C1C(=O)O)O)C2C=C(CCC2C(=C)C)C)O","CC1=CC(C(CC1)C(=C)C)C2=C(C=C(C=C2O)C)O",
           "CC1=C[C@@H](C2=C(O)C=C(CCCC)C=C2O)[C@H](C(C)=C)CC1","CCCC1=CC(=C(C(=C1)O)C2C=C(CCC2C(=C)C)C)O",
           "CCCC1=CC(=C(C(=C1C(=O)O)O)C2C=C(CCC2C(=C)C)C)O","OC(C=C(C=C1O)CCCCCCC)=C1[C@H]2[C@H](C(C)=C)CCC(C)=C2",
           "OC(C=C(C=C1O)CCCCC)=C1[C@H]2[C@H](C(C)=C)C[C@H](O)C(C)=C2","CCCCCC1=CC(=C2C3C(CCC(C3OC2=C1C(=O)O)(C)O)C(=C)C)O",
           "CCCCCC1=CC(=C2C3C(CCC(C3OC2=C1)(C)O)C(=C)C)O","CCCCCC1=CC(=C(C(=C1)O)CC=C(C)CCC=C(C)C)O",
           "CC1=CC2=C(C(C)(C)OC3=C2C(OC)=CC(CCCCC)=C3)C=C1","CCCCCC1=CC(=C(C(=C1C(=O)O)O)CC=C(C)CCC=C(C)C)O",
           "CCCCCC1=CC(=C(C(=C1C(=O)O)O)CC=C(C)CCC=C(C)C)OC","CCCC1=CC(=C(C(=C1)O)CC=C(C)CCC=C(C)C)O",
           "CCCC1=CC(=C(C(=C1C(=O)O)O)CC=C(C)CCC=C(C)C)O","OC1=C(C/C=C(C)/CC/C=C(C)/C)C(O)=CC(C)=C1",
           "OC1=C(C/C=C(C)/CC/C=C(C)/C)C(O)=C(C(O)=O)C(C)=C1","CCCCCC1=CC(=C(C(=C1)O)C2=C(C=CC(=C2)C)C(=C)C)O",
           "CCCCCC1=CC(=C2C(=C1)OC(C3=C2C=C(C=C3)C)(C)C)O","CCCCCC1=CC2=C(C3=C(C=CC(=C3)C)C(O2)(C)C)C(=C1C(=O)O)O",
           "CCCC1=CC(=C2C(=C1)OC(C3=C2C=C(C=C3)C)(C)C)O","CCCCCC1=CC(=C2C(=C1)OC(C3=C2C(C(CC3)(C)O)O)(C)C)O",
           "CCCCCC1=CC(=C2C3CC(=CCC3C(OC2=C1)(C)C)C)O","CC1=CC[C@@]2([H])C(C)(C)OC(C=C(CCCCC)C(C(O)=O)=C3O)=C3[C@]2([H])C1",
           "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O","CCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O",
           "CCCCCC1=CC2=C(C3C=C(CCC3C(O2)(C)C)C)C(=C1C(=O)O)O","CCCCC1=CC2=C([C@@H]3C=C(CC[C@H]3C(O2)(C)C)C)C(=C1)O",
           "CC1=CC2C(CC1)C(OC3=CC(=CC(=C23)O)C)(C)C","CC1=CC2C(CC1)C(OC3=C2C(=C(C(=C3)C)C(=O)O)O)(C)C",
           "CC1=CC2C(CC1)C(OC3=C(C(=CC(=C23)O)C)C(=O)O)(C)C","CCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O",
           "CCCC1=CC2=C(C3C=C(CCC3C(O2)(C)C)C)C(=C1C(=O)O)O","CC1=C[C@]2([H])[C@@](C(C)(C)OC3=C2C(O)=CC(CCCCCCC)=C3)([H])CC1",
           "CCCCCC1=CC(=C2C(=C1)OC3=C(C=CC(=C23)C(C)C)C)O","CCCCCC1=CC(=C2C3C(CCC(C3O)(C)O)C(OC2=C1)(C)C)O",
           "CCCCCC1=CC2=C3C(OC(C)(C)C4C3CC(C)(CC4)O2)=C1","CCCCCC1=CC(=C2C(=C1)OC3=C(C=CC(=C23)C(=C)C)C)O",
           "CCCCCC1=CC(=C2C3C=C(CCC3C(OC2=C1)(C)C)C)O")
results_all <- data.frame(matrix("",nrow=0,ncol=4))
colnames(results_all) <- c("name","smile","downregulated","upregulated")
for (i in 1:length(files)) {
  file <- read.csv(files[i])
  
  if (nrow(file)<5) {
    df_results <- data.frame(matrix("",nrow=1,ncol=4))
    colnames(df_results) <- c("name","smile","downregulated","upregulated")
    df_results$name <- names[i]
    df_results$smile <- smiles[i]
    df_results$downregulated <- ""
    df_results$upregulated <- ""
    
  }else {
    
    results <- infer_pathways(cell_line = cell,path_pa = path_pa,neighbor_file = files[i],all = all,n_min_neighbors = 5,
                                   tb_pa = 10,dist_pb = dist_pb,dist_pt = dist_pt,fr_thresh = 0.65,p_val_thresh = 0.01,n_trials = 5000)
    df_results <- data.frame(matrix("",nrow=10,ncol=4))
    colnames(df_results) <- c("name","smile","downregulated","upregulated")
    df_results$name <- names[i]
    df_results$smile <- smiles[i]
    len_down <- length(results[[1]]$Downregulated)
    len_up <- length(results[[1]]$Upregulated)
    df_results$downregulated <- as.character(df_results$downregulated)
    df_results$upregulated <- as.character(df_results$upregulated)
    df_results$downregulated[1:len_down] <- as.character(results[[1]]$Downregulated)
    df_results$upregulated[1:len_up] <- results[[1]]$Upregulated
  }
  results_all <- bind_rows(results_all,df_results)
}
write.csv(results_all,"C:/Users/user/Documents/deepSIBA/results/cannabis_pathways_a375/a375_cannabis_pathways.csv",row.names = F)


