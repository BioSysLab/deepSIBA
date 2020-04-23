library(AnnotationDbi)
library(tidyverse)

##Load paths for training sets
train_path <- c("C:/Users/user/Documents/deepSIBA/learning/data/vcap/train_test_split/train.csv",
                "C:/Users/user/Documents/deepSIBA/learning/data/mcf7/train_test_split/train.csv",
                "C:/Users/user/Documents/deepSIBA/learning/data/a375/train_test_split/train.csv",
                "C:/Users/user/Documents/deepSIBA/learning/data/pc3/train_test_split/train.csv")

cell <- c('vcap','mcf7','a375','pc3') 

#Output folder path pattern
out <- "C:/Users/user/Documents/deepSIBA/article_results"

#Input ranging scale of go term distance (1 or 2)
sc <- 2

for (i in 1:length(cell)){
  train <- read.csv(train_path[i])
  png(paste0(out,"_",cell[i],"/",cell[i],"_train.png"), width = 8, height = 6,units="in",res=300)
  hist(train$value/sc,main=toupper(cell[i]),xlab = "GO term distance",col="gray")
  dev.off()
  gc()
}


