library(tidyverse)

all <- read.csv("C:/Users/user/Documents/deepSIBA/learning/data/vcap/alldata/alldata_vcap.csv")

all_drugs <- unique(c(as.character(all$Var1),as.character(all$Var2)))

pert_info <- read.delim("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/cmap/GSE92742_Broad_LINCS_pert_info.txt")
all_drugs <- as.data.frame(all_drugs)
all_drugs <- left_join(all_drugs,pert_info,by = c("all_drugs"="pert_id"))

all_drugs <- all_drugs %>% filter(is_touchstone == 1)
all$value <- all$value/2

touch <- 0
for (i in 1:nrow(all_drugs)) {
  
  ind <- unique(c(which(all$Var1 %in% all_drugs$all_drugs[i]),which(all$Var2 %in% all_drugs$all_drugs[i])))
  filt <- all[ind,]
  filt<-filt[order(filt$value),]
  touch[i] <- (filt$value[92])
  
}

touch <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_8/hist_vcap.rds")

png(file="supl_fig_8vcap.png",width=8,height=6,units = "in",res=300)
hist(touch,breaks = seq(0,0.5,0.02), xlab = "Threshold equivalent of 90% cmap score", col = "grey",main = NULL)
title("VCAP",adj = 0)
abline(v=0.22, col = "red", lwd = 2)
abline(v=0.20, col = "red", lwd = 2)
dev.off()
