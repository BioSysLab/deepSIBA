cross_validation <- function(dataframe,allq1smiles,ecfp_sims,no_folds,n_drugs,max_sim,min_sim,dir){
  # dataframe has all training data with 0-2 distances
  # allsmiles should be character vector of all rdkit smiles
  # ecfp sims is the square matrix of ecfp similarities from all_smiles
  library(tidyverse)
  ecfp_sims <- ecfp_sims[,-1]
  ecfp_sims <- as.matrix(ecfp_sims)
  diag(ecfp_sims) <- 0
  colmax <- apply(ecfp_sims,2,max)
  indcandidates <- which(colmax < max_sim & colmax >= min_sim)
  allq1smiles <- as.character(allq1smiles$x)
  names(colmax) <- allq1smiles
  #hist(colmax[indcandidates])
  candidates <- as.character(allq1smiles[indcandidates])
  breaks <- c(0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9)
  cold_val <- function(histogram,breaks,all_data,drug_vector,ncold,mtry){
    library(tidyverse)
    library(MLmetrics)
    
    best_mse <- 1
    for (i in 1:mtry) {
      drugs_sampled <- sample(drug_vector,ncold,replace = F)
      indices_all <- unique(union(which(all_data$rdkit.x %in% drugs_sampled),which(all_data$rdkit.y %in% drugs_sampled)))
      hist_new <- hist(all_data[indices_all,]$value,breaks = breaks,freq = T,plot = F)
      mse <- MSE(hist_new$density,histogram$density)
      if (mse < best_mse) {
        best_mse <- mse
        best_sample <- drugs_sampled
        print(best_mse)
        print(i)
      }
    }
    return(best_sample)
  }
  a <- hist(dataframe$value,breaks = breaks)
  for (i in 1:no_folds) {
    alldata <- dataframe
    cold <- cold_val(histogram = a,breaks = breaks,all_data = alldata,drug_vector = as.character(candidates),ncold = n_drugs,mtry = 20000)
    alldata$iscoldx <- alldata$rdkit.x %in% cold
    alldata$iscoldy <- alldata$rdkit.y %in% cold
    
    val_data <- alldata %>% filter(iscoldx == T | iscoldy == T)
    train_data <- anti_join(alldata,val_data)
    
    dir.create(paste0(dir,"/fold_",i))
    png(filename = paste0(dir,"/fold_",i,"/cold_hist_",i,".png"))
    hist_cold <- hist(val_data$value,breaks = breaks,freq = T)
    dev.off()
    
    trainsmiles <- unique(c(train_data$rdkit.x,train_data$rdkit.y))
    
    write.csv(cold,paste0(dir,"/fold_",i,"/valsmiles_",i,".csv"))
    write.csv(trainsmiles,paste0(dir,"/fold_",i,"/trainsmiles_",i,".csv"))
    
    indkeep <- unique(c(which(val_data$rdkit.x %in% cold),which(val_data$rdkit.y %in% cold)))
    
    val_data <- val_data[indkeep,]
    val_data_cold <- val_data %>% filter((iscoldx == T & iscoldy == T))
    val_data <- val_data %>% filter(!(iscoldx == T & iscoldy == T))
    write.csv(train_data,paste0(dir,"/fold_",i,"/train_",i,".csv"))
    write.csv(val_data,paste0(dir,"/fold_",i,"/val_",i,".csv"))
    write.csv(val_data_cold,paste0(dir,"/fold_",i,"/val_cold_",i,".csv"))
    
    # make new candidates
    print(length(which(cold %in% trainsmiles)))
    candidates <- candidates[-which(candidates %in% cold)]
    
    png(filename = paste0(dir,"/fold_",i,"/val_sims_",i,".png"))
    hist_sims <- hist(colmax[cold],breaks = seq(0,1,0.1),freq = T)
    dev.off()
    
  }
  
  
}
