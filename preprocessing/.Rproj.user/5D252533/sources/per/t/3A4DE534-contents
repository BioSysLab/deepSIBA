library(tidyverse)
library(doFuture)

# parallel set number of workers
registerDoFuture()
plan(multiprocess,workers = 12)

# read the required files

sig_info <- read.delim("data_preprocessing/cmap/GSE92742_Broad_LINCS_sig_info.txt")
sig_metrics <- read.delim("data_preprocessing/cmap/GSE92742_Broad_LINCS_sig_metrics.txt")
# the full path to the cmap gctx file
ds_path <- "C:/Users/user/Documents/phd/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
landmark <- read_tsv(file = "data_preprocessing/cmap/cmap_landmark_genes.txt")
go_anno <- readRDS("data_preprocessing/utility/goterm_annotation.rds")
pert <- readRDS("data_preprocessing/utility/pert_id_to_rdkit.rds")

# get the signatures for a375
a375 <- drug_sigs_per_line(cell_line = "A375",sig_info = sig_info,
                           sig_metrics = sig_metrics)

# keep quality 1 signatures and compounds with rdkit smiles

a375 <- a375 %>% filter(quality==1)
a375 <- a375[which(as.character(a375$pert_id) %in% pert$pert_id),]

# calculate enriched go terms

go_nes <- go_path_analysis(sig_ids = a375$sig_id,cmap_path_to_gctx = ds_path,landmark_df = landmark,goterms = go_anno)
a375_go <- go_nes[[1]]
# calculate pairwise distances with ensemble approach and average

# run distances
thresholds <- c(10,20,30,40,50)

dist_a375 <- NULL
### calculate distances
dist_a375 <- foreach(thres = thresholds) %dopar% {
  distance_scores(num_table = a375_go,threshold_count = thres,names = as.character(a375$pert_id))
}

# average distance for different thresholds
distance <- do.call(cbind,dist_a375)
distance <- array(distance,c(dim=dim(dist_a375[[1]]),length(dist_a375)))
mean_dist_a375 <- apply(distance, c(1,2), mean, na.rm = TRUE)

mean_dist_a375[lower.tri(mean_dist_a375,diag = T)] <- 666

colnames(mean_dist_a375) <- as.character(a375$pert_id)
rownames(mean_dist_a375) <- as.character(a375$pert_id)
# long format dataframe
a375_dist <- reshape2::melt(mean_dist_a375)

a375_dist <- a375_dist %>% filter(value != 666)
# add rdkit smiles to a375
a375 <- left_join(a375,pert)
a375 <- a375 %>% dplyr::select(sig_id,pert_id,pert_iname,quality,rdkit,pert_dose,pert_time)

# add info from the a375 dataframe
a375_dist <- left_join(a375_dist,a375, by = c("Var1"="pert_id"))
a375_dist <- left_join(a375_dist,a375, by = c("Var2"="pert_id"))

#extract test train split
# tanimoto similarities between ECFP4 fingerprints of the rdkit compounds
ecfp_sims <- read.csv("data_preprocessing/utility/sims/a375q1similarities.csv")
allq1smiles <- read.csv("data_preprocessing/utility/smiles/alla375q1smiles.csv")
# input distance dataframe
dataframe <- a375_dist
dir <- "output_dir"
# set no_folds to 5 for 5 fold cv split
no_folds <- 1
# n_drugs number of drugs in the test/val set
n_drugs <- 80
# maximum and minimum allowed similarity between test/val and training set compounds
max_sim <- 0.8
min_sim <- 0.0
# extract split
cross_validation(dataframe = dataframe,allq1smiles = allq1smiles,ecfp_sims = ecfp_sims,no_folds = 5,n_drugs = 80,max_sim = 0.9,
                 min_sim = 0.0,dir = dir)
