library(tidyverse)
# read training set of mcf7 with BP and ecfp4 distances
new <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_1/train_with_ecfp4.csv")

new$value <- new$value/2
# read gene level distances
genes <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/table_extra_data/table_1/gene_distances.rds")

new <- left_join(new,genes,by=c("Var1"="Var1","Var2"="Var2"))


dist01 <- new %>% filter(dist <= 0.1)

thresh <- seq(0,1,0.1)
ppg <- NULL
ppbp <- NULL
for (i in 1:length(thresh)) {
  ng <- length(which(dist01$value.y<=thresh[i]))
  nb <- length(which(dist01$value.x<=thresh[i]))
  ppg[i] <- ng/nrow(dist01)
  ppbp[i] <- nb/nrow(dist01) 
}

plot(thresh,ppbp,type = "l")
lines(thresh,ppg,col = "red")
