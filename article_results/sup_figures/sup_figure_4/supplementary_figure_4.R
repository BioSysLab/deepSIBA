library(tidyverse)
library(doFuture)
library(cancerTiming)
drug_sigs_per_line_dups <- function(cell_line,sig_info,sig_metrics) {
  
  # cell_line character of cell line
  # sig_info dataframe of GSE info
  # sig_metrics dataframe of GSE metrics
  
  library(tidyverse)
  options(warn =- 1)
  
  
  cell <- sig_info %>%
    filter(cell_id == cell_line) %>%
    filter(pert_type == "trt_cp") %>%
    group_by(pert_iname) %>%
    mutate(count = n_distinct(sig_id)) %>%
    ungroup()
  
  print(paste0('the unique drugs for ',cell_line,' are ',length(unique(cell$pert_iname))))
  
  ## add the signature metrics
  
  cell <- left_join(cell,sig_metrics)
  
  ## keep the drugs that we have only 1 signature for this cell line
  
  cell_singles <- cell %>%
    filter(count == 1) %>%
    dplyr::select(-count)
  
  print(paste0('the drugs that have only 1 signature for ',cell_line,' are ',length(unique(cell_singles$pert_iname))))
  
  cell_singles$pert_itime <- factor(cell_singles$pert_itime)
  print("time summary")
  print(summary(cell_singles$pert_itime))
  cell_singles$pert_idose <- factor(cell_singles$pert_idose)
  print("dose summary")
  print(summary(cell_singles$pert_idose))
  
  ## add quality column to single perturbations
  
  cell_singles$quality <- 100
  
  cell_singles <- cell_singles %>%
    mutate(quality = if_else(is_exemplar == 1 & tas > 0.4 & distil_nsample>=2 ,true = 1,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.2 & tas<=0.4 & distil_nsample>2 ,true = 2,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.2 & tas<=0.4 & distil_nsample <=2 ,true = 3,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.1 & tas<=0.2 & distil_nsample>2 ,true = 4,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.1 & tas<=0.2 & distil_nsample <= 2 ,true = 5,false = quality),
           quality = if_else(is_exemplar == 1 & tas < 0.1 & distil_nsample > 2 ,true = 6,false = quality),
           quality = if_else(is_exemplar == 1 & tas < 0.1 & distil_nsample <= 2 ,true = 7,false = quality),
           quality = if_else(is_exemplar == 0 ,true = 8,false = quality),
           quality = factor(quality))
  
  print("summary of the quality of drugs with only 1 signature")
  print(summary(cell_singles$quality))
  
  ## keep the multiple signature drugs in cell
  
  cell<- anti_join(cell,cell_singles)
  
  ### add priorities to the multiple signatures
  
  cell$priority <- 100
  cell <- cell %>%
    mutate(priority = if_else(pert_dose == "10.0" & pert_time == 24,true = 1,false = priority),
           priority = if_else(pert_idose == "5 ÂµM" & pert_time == 24,true = 2,false = priority),
           priority = if_else(pert_idose != "5 ÂµM" & pert_dose != "10.0" & pert_time == 24,true = 3,false = priority),
           priority = if_else(pert_dose == "10.0" & pert_time == 6,true = 4,false = priority),
           priority = if_else(pert_idose == "5 ÂµM" & pert_time == 6,true = 5,false = priority),
           priority = if_else(pert_idose != "5 ÂµM" & pert_dose != "10.0" & pert_time == 6,true = 6,false = priority),
           priority = factor(priority))
  
  print("priorities for drugs with multiple signatures")
  print(summary(cell$priority))
  ### add quality to the multiple signatures
  
  cell$quality <- 100
  
  cell <- cell %>%
    mutate(quality = if_else(is_exemplar == 1 & tas > 0.4 & distil_nsample>=2 ,true = 1,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.2 & tas<=0.4 & distil_nsample>2 ,true = 2,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.2 & tas<=0.4 & distil_nsample <=2 ,true = 3,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.1 & tas<=0.2 & distil_nsample>2 ,true = 4,false = quality),
           quality = if_else(is_exemplar == 1 & tas > 0.1 & tas<=0.2 & distil_nsample <= 2 ,true = 5,false = quality),
           quality = if_else(is_exemplar == 1 & tas < 0.1 & distil_nsample > 2 ,true = 6,false = quality),
           quality = if_else(is_exemplar == 1 & tas < 0.1 & distil_nsample <= 2 ,true = 7,false = quality),
           quality = if_else(is_exemplar == 0 ,true = 8,false = quality),
           quality = factor(quality))
  
  
  print("summary of the quality of drugs with multiple signatures")
  print(summary(cell$quality))
  
  
  print(paste0('the drugs that have Multiple signatures for ',cell_line,' are ',length(unique(cell$pert_iname))))
  
  
  
  
  #### clean them based on quality for each drug and then solve the equalities with max tas
  
  
  
  
  cell_cleaned <- cell %>%
    group_by(pert_iname) %>%
    filter(quality == min(as.numeric(quality))) %>%
    ungroup %>%
    dplyr::select(-c(count,priority))
  
  
  
  cell_final <- bind_rows(cell_cleaned,cell_singles)
  
  print("summary of final quality of signatures")
  print(summary(cell_final$quality))
  
  return(cell_cleaned)
}
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
sig <- read.delim("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/cmap/GSE92742_Broad_LINCS_sig_info.txt")
sig_metrics <- read.delim("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/cmap/GSE92742_Broad_LINCS_sig_metrics.txt")
gos <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/utility/goterm_annotation.rds")
thres <- c(10,20,30,40,50)


reps_info <- drug_sigs_per_line_dups(cell_line = "MCF7",sig_info = sig,sig_metrics = sig_metrics)
reps_info$quality <- as.character(reps_info$quality)
reps_info <- reps_info %>% filter(quality == "1") %>% 
  group_by(pert_iname) %>%
  mutate(count = n_distinct(sig_id),
         count_dose = n_distinct(pert_dose),
         count_time = n_distinct(pert_time)) %>% 
  ungroup() %>% filter(count > 1) %>% filter(count_dose == 1) %>% filter(count_time == 1)

drugs <- unique(as.character(reps_info$pert_iname))
rep_mean_q1 <- matrix(0,nrow = length(drugs),ncol = 1)
rownames(rep_mean_q1) <- drugs

for (i in 1:length(drugs)) {
  drug <- drugs[i]
  dist_rep <- list(0)
  paths <- go_path_analysis(sig_ids = reps_info$sig_id[which(reps_info$pert_iname %in% drug)],cmap_path_to_gctx = ds_path,landmark_df = landmark, goterms = gos)
  nes2 <- paths[[1]]
  for (j in 1:length(thres)) {
    dist2 <- distance_scores(num_table = nes2,threshold_count = thres[j],names = colnames(nes2))
    dist_rep[[j]] <- dist2
  }
  sam <- 0
  for (k in 1:length(dist_rep)) {
    sam <- sam + dist_rep[[k]]
  }
  sam <- sam/length(dist_rep)
  rep_mean_q1[i,1] <- mean(sam[upper.tri(x = sam,diag = F)])
  #rep_mean[i,1] <- sam
}

mcf7_q1_dups_go <- rep_mean_q1/2
mcf7_q1_random <- sample_n(mcf7_go,100)

png(file="supl_fig_4.png",width=8,height=6,units = "in",res=300)
multidensity(list(mcf7_q1_dups_go,mcf7_q1_random$value),xlim = c(0,1),main= NULL,xlab = "Ensemble GO term distance",
             xaxs = "i", yaxs = "i")
legend("topright", 
       legend = c("Q1 duplicate pairs (n = 20)", "Q1 random pairs (n = 100)"), 
       col = c('black', 
               'red'),
       lty = c(1,1),
       bty = "o", 
       pt.cex = 1.5, 
       cex = 0.8, 
       text.col = "black", 
       horiz = F , 
       inset = c(0.01, 0.01))
dev.off()