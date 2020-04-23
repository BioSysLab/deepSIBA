library(tidyverse)
library(cmapR)
library(rhdf5)
library(AnnotationDbi)
library(org.Hs.eg.db)

# load the GO library
library(GO.db)
library(doFuture)

registerDoFuture()
plan(multiprocess,workers = 12)
# extract a named vector of all terms
goterms <- Term(GOTERM)

get_signatures_tb_commons <- function(gos,tb_go,sig_ids) {
  
  
  col1 <- grep(sig_ids[1],colnames(gos))
  col2 <- grep(sig_ids[2],colnames(gos))
  top1 <- rownames(gos)[order(gos[,col1])][nrow(gos):(nrow(gos)-tb_go)]
  bot1 <- rownames(gos)[order(gos[,col1])][1:tb_go]
  top2 <- rownames(gos)[order(gos[,col2])][nrow(gos):(nrow(gos)-tb_go)]
  bot2 <- rownames(gos)[order(gos[,col2])][1:tb_go]
  com_bot_gos <- bot1[which(bot1 %in% bot2)]
  com_top_gos <- top1[which(top1 %in% top2)]
  #Return output list
  cor <- cor(gos[,col1],gos[,col2])
  output <- list(length(com_bot_gos)+length(com_top_gos),cor)
  return(output)
}

mcf7gos <- readRDS("MCF7_go.rds")
mcf7gos <- mcf7gos[[1]]

## run for the train data
## add sig id to train

line <- line[,c("sigs","pert_id")]

train <- left_join(train,line,by = c("Var1"="pert_id"))
colnames(line)<-c("sigs2","pert_id")
train <- left_join(train,line,by = c("Var2"="pert_id"))
train$value <- train$value/2
train$sigs <- as.character(train$sigs)
train$sigs2 <- as.character(train$sigs2)

wrapper <- function(df,threshold,num){
  threshold2 <- threshold - 0.01
  df2<-df %>%
    filter(value<=threshold) %>% filter(value > threshold2)
  s<-matrix(0,nrow = nrow(df2),ncol=1)
  for (i in 1:nrow(df2)) {
    
    s[i,1]<-get_signatures_tb_commons(gos = mcf7gos,tb_go = num,sig_ids = c(df2$sigs[i],df2$sigs2[i]))
    
  }
  return(s)
}


thresholds <- seq(0.14,0.22,0.01)
all25 <- list(0)
for (i in 1:length(thresholds)) {
  all25[[i]] <- wrapper(df = train,threshold = thresholds[i],num = 25)
  print(i)
  
}

all25 <- foreach(thres = thresholds) %dopar% {
  wrapper(df = train[1:50000,],threshold = thresh)
}

get_signatures_tb_commons(gos = mcf7gos,tb_go = 25,sig_ids = filt1$sig_id[1:2])

ten <-0
tobind <-data[,1:16]
colnames(tobind) <- colnames(train)
tobind$value <- tobind$value * 2
all <- bind_rows(train,rest,tobind)
drugs <- unique(c(as.character(all$Var1),as.character(all$Var2)))

for (i in 1:length(drugs)) {
  
  ind <- unique(c(which(all$Var1 %in% drugs[i]),which(all$Var2 %in% drugs[i])))
  filt <- all[ind,]
  filt<-filt[order(filt$value),]
  ten[i] <- (filt$value[342])
  print(i)
}

ten <- ten / 2
hist(ten,breaks = seq(0,1,0.02))
# precision for each drug
prec <- function(df,thresh){
  tpdf <- df %>% filter(pred<=thresh) %>% filter(true<=thresh)
  tp <- nrow(tpdf)
  pos <- length(which(df$pred<=thresh))
  precision <- tp/pos
  return(precision)
}

#threshold for each drug

drugs <- unique(c(as.character(data$Var1),as.character(data$Var2)))

qq<-0
for (i in 1:length(drugs)) {
  
  ind <- unique(c(which(data$Var1 %in% drugs[i]),which(data$Var2 %in% drugs[i])))
  filt<-data[ind,]
  filt <- filt[order(filt$pred),]
  qq[i] <- (prec(df = filt,thresh = 0.18))
  
  
}
hist(qq,xlab="precision",breaks = seq(0,1,0.05),ylim = c(0,200),main = "histogram of precisions for each drug in the query @0.18 threshold")

qq_resim<-0
for (i in 1:length(drugs)) {
  
  ind <- unique(c(which(resim2$Var1 %in% drugs[i]),which(resim2$Var2 %in% drugs[i])))
  filt<-resim2[ind,]
  filt <- filt[order(filt$pred),]
  qq_resim[i] <- (prec(df = filt,thresh = 0.18))
  
  
}
hist(qq_resim,xlab="precision",breaks = seq(0,1,0.05),ylim = c(0,200),main = "histogram of precisions for each drug in the query @0.18 threshold Resimnet")

qq_chris<-0
for (i in 1:length(drugs)) {
  
  ind <- unique(c(which(mydata$Var1 %in% drugs[i]),which(mydata$Var2 %in% drugs[i])))
  filt<-mydata[ind,]
  qq_chris[i] <- (prec(df = filt,thresh = 0.18))
  
  
}
datanick <- data
datanick$pred <- data2$V1

qq_nick <- 0

for (i in 1:length(drugs)) {
  
  ind <- unique(c(which(datanick$Var1 %in% drugs[i]),which(datanick$Var2 %in% drugs[i])))
  filt<-datanick[ind,]
  qq_nick[i] <- (prec(df = filt,thresh = 0.22))
  
  
}

sim10 <- 0
for (i in 1:length(thresholds)) {
  
  sim10[i] <- mean(apply(all[[i]],MARGIN = 1,sum))
  
}

sim25 <- 0

for (i in 1:length(thresholds)) {
  
  sim25[i] <- mean(all25[[i]])
  
}

sim10 <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_7/top_10_gos.rds")
sim25 <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/supplementary_figure_data/supplementary_figure_7/top_25_gos.rds")
thresholds <- seq(0.14,0.22,0.01)


png(file="supl_fig_7.png",width=8,height=6,units = "in",res = 300)
plot(thresholds,sim25,ylim = c(5,12),"o",xlim = c(0.139,0.221),col = "black",xlab = "Threshold",
     ylab = "Number of common GO terms" ,lwd = 1.7,lty = 5,pch = 1,yaxs="i",xaxs = "i")
dev.off()