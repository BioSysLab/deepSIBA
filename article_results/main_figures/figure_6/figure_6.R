library(tidyverse)
library(MLmetrics)

prec_eval <- function(true,pred,thresh){
  dd <- cbind(true,pred)
  dd <- as.data.frame(dd)
  colnames(dd) <- c("true","pred")
  tp <- length(which(dd$true<=thresh & dd$pred<=thresh))
  pos <- length(which(dd$pred<=thresh))
  prec <- tp/pos
  return(prec)
}
a3752 <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/figure_extra_data/figure_6/val_fixed.csv")
mu <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/figure_extra_data/figure_6/mu_star3.csv",header = F)
sigma <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/figure_extra_data/figure_6/sigma_star3.csv",header = F)

a3752$V1 <- mu$V1
a3752$value <- a3752$value/2
a3752$sigma <- sigma$V1
a3752$cv <- a3752$sigma/a3752$V1

thresh_pr_a3752 <- seq(0.15,0.25,0.01)
prec_a3752 <- NULL
n_a3752 <- NULL
#ms_a3752 <- NULL
for (i in 1:length(thresh_pr_a3752)) {
  filt <- a3752 %>% filter(cv <= thresh_pr_a3752[i])
  prec_a3752[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.22)
  n_a3752[i] <- length(which(filt$V1<=0.22))
  #ms_a3752[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
}

plot(thresh_pr_a3752,prec_a3752)

thresh_ms_a3752 <- c(seq(0.12,0.15,0.005),seq(0.16,0.25,0.01))
ms_a3752 <- NULL
for (i in 1:length(thresh_ms_a3752)) {
  filt <- a3752 %>% filter(cv <= thresh_ms_a3752[i])
  #prec_a3752[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.22)
  #n_a3752[i] <- length(which(filt$V1<=0.22))
  ms_a3752[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
}


thresh_cor_a3752 <- seq(0.12,0.25,0.01)
cor_a3752 <- NULL
for (i in 1:length(thresh_cor_a3752)) {
  filt <- a3752 %>% filter(cv <= thresh_cor_a3752[i])
  #prec_a3752[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.22)
  #n_a3752[i] <- length(which(filt$V1<=0.22))
  cor_a3752[i] <- cor(filt$V1,filt$value)
}



pdf(file="result_fig4b.pdf",width=7,height=6)
png(file="result_fig4b.png",width=7,height=6,units = "in",res = 300)
plot(thresh_pr_a3752,prec_a3752,ylim = c(0,1.1),"o",xlim = c(0.15,0.25),xlab = "CV threshold",
     ylab = "Precision" ,lwd = 1.5,lty = 2,pch = 3,cex = 1,yaxs = "i", col = "black")
title("B", adj = 0)
dev.off()


plot(thresh_ms_a3752,ms_a3752,ylim = c(0,0.012),"o",xlim = c(0.12,0.25),col = "#4daf4a",xlab = "CV threshold",
     ylab = "Precision" ,lwd = 1.5,lty = 2,pch = 3,cex = 1,main = "Precision for different uncertainty thresholds ")

### result figure 4a 
### fix the old val data
sims <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/figure_extra_data/figure_6/train_val_sims.csv")
sims <- sims[,-1]

sims_cat <- sims >0.3 
sims_cat <- sims_cat+0
cold <- read.csv("C:/Users/user/Documents/deepSIBA/article_results/figure_extra_data/figure_6/valsmiles.csv")
cold <- as.character(cold$x)
keep <- cold[which(colSums(sims_cat) == 0)]
png(filename = "result_fig4a.png",width = 7,height = 6,units = "in",res = 300)
hist(apply(sims[,which(colSums(sims_cat) == 0)],2,max),col = "grey",xlim = c(0,1),ylim = c(0,25),xlab = "Maximum structural similarity",main = NULL)
title("A", adj = 0)
dev.off()