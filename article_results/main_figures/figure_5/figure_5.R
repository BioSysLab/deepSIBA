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

mcf7 <- read.csv("C:/Users/user/Documents/deepSIBA/learning/data/mcf7/train_test_split/test.csv")

thresh <- seq(0.25,0.95,0.1)
cors <- NULL
ms_mcf7 <- NULL
prec_mcf7 <- NULL
for (i in 1:length(thresh)) {
  filt <- mcf7 %>% filter(dist >= (thresh[i]-0.05)) %>% filter(dist <= (thresh[i]+0.05))
  ms_mcf7[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
  #cors[i] <- cor(x = filt$V1,y = filt$value)
  prec_mcf7[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
}

vcap <- read.csv("C:/Users/user/Documents/deepSIBA/learning/data/vcap/train_test_split/test.csv")
ms_vcap <- NULL
prec_vcap <- NULL
for (i in 1:length(thresh)) {
  filt <- vcap %>% filter(dist >= (thresh[i]-0.05)) %>% filter(dist <= (thresh[i]+0.05))
  ms_vcap[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
  #cors[i] <- cor(x = filt$V1,y = filt$value)
  prec_vcap[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
}

a375 <- read.csv("C:/Users/user/Documents/deepSIBA/learning/data/a375/train_test_split/test.csv")
ms_a375 <- NULL
prec_a375 <- NULL
for (i in 1:length(thresh)) {
  filt <- a375 %>% filter(dist >= (thresh[i]-0.05)) %>% filter(dist <= (thresh[i]+0.05))
  ms_a375[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
  #cors[i] <- cor(x = filt$V1,y = filt$value)
  prec_a375[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
}

pc3 <- read.csv("C:/Users/user/Documents/deepSIBA/learning/data/pc3/train_test_split/test.csv")
ms_pc3 <- NULL
prec_pc3 <- NULL
for (i in 1:length(thresh)) {
  filt <- pc3 %>% filter(dist >= (thresh[i]-0.05)) %>% filter(dist <= (thresh[i]+0.05))
  ms_pc3[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
  #cors[i] <- cor(x = filt$V1,y = filt$value)
  prec_pc3[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
}


pdf(file="result_fig3a.pdf",width=7,height=6)
png(file="result_fig3a.png",width=7,height=6,units = "in",res = 300)
plot(thresh,ms_pc3,ylim = c(0,0.045),"o",xlim = c(0.247,0.953),col = "black",xlab = "Structural distance range",
     ylab = "MSE" ,lwd = 1,xaxt = "n",lty = 5,pch = 1,,xaxs = "i",yaxs="i",cex = 0.6)
axis(1, at=seq(0.25,0.95,0.1), labels=c("[0.2,0.3]","[0.3,0.4]","[0.4,0.5]",
                                        "[0.5,0.6]","[0.6,0.7]","[0.7,0.8]","[0.8,0.9]","[0.9,1.0]"),cex.axis = 0.8,xaxs = "i")
lines(thresh,ms_mcf7,col = "#e41a1c",lwd = 1.7,type = "o",lty = 5,pch = 2,cex = 0.6)
lines(thresh,ms_a375,col = "#4daf4a",lwd = 1.7, type = "o",lty = 2 , pch = 3,cex = 0.6)
lines(thresh,ms_vcap,col = "#377eb8",lwd = 1.7, type = "o",lty = 5,pch = 4,cex = 0.6)
title("A",adj = 0)
legend("topleft", 
       legend = c("PC3", "VCAP", "A375","MCF7"),
       col = c('black', 
               '#377eb8',"#4daf4a","#e41a1c"),pch =c(1,4,3,2),
       pt.cex = 0.6, lwd = 0.8,
       cex = 0.6, 
       text.col = "black", 
       horiz = F )
dev.off()




pdf(file="result_fig3b.pdf",width=7,height=6)
png(file="result_fig3b.png",width=7,height=6,units = "in",res = 300)
plot(thresh,prec_pc3,ylim = c(0,1.1),"o",xlim = c(0.25,0.95),col = "black",xlab = "Structural distance range",
     ylab = "Precision" ,lwd = 1.7,xaxt = "n",lty = 5,pch = 1,yaxs="i",cex = 0.6)
lines(thresh,c(NaN,1,1,1,1,NaN,NaN,NaN),col = "black",lwd = 1.7,type = "l",lty = 5)
axis(1, at=seq(0.25,0.95,0.1), labels=c("[0.2,0.3]","[0.3,0.4]","[0.4,0.5]",
                                        "[0.5,0.6]","[0.6,0.7]","[0.7,0.8]","[0.8,0.9]","[0.9,1.0]"),cex.axis = 0.8,xaxs = "i")
lines(thresh,prec_mcf7,col = "#e41a1c",lwd = 1.7,type = "o",lty = 5,pch = 2,cex = 0.6)
lines(thresh,prec_a375,col = "#4daf4a",lwd = 1.7, type = "o",lty = 2 , pch = 3,cex = 0.6)
lines(thresh,prec_vcap,col = "#377eb8",lwd = 1.7, type = "o",lty = 5,pch = 4,cex = 0.6)
title("B",adj = 0)
legend("topleft", 
       legend = c("PC3", "VCAP", "A375","MCF7"),
       col = c('black', 
               '#377eb8',"#4daf4a","#e41a1c"),pch =c(1,4,3,2),
       pt.cex = 0.6, lwd = 0.8,
       cex = 0.6, 
       text.col = "black", 
       horiz = F )
dev.off()


mcf7$cv <- mcf7$sigma_star/mcf7$V1
thresh_pr_mcf7 <- seq(0.23,0.4,0.01)
prec_mcf7 <- NULL
n_mcf7 <- NULL
for (i in 1:length(thresh_pr_mcf7)) {
  filt <- mcf7 %>% filter(cv <= thresh_pr_mcf7[i])
  prec_mcf7[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
  n_mcf7[i] <- length(which(filt$V1<=0.2))
}

thresh_ms_mcf7 <- c(seq(0.14,0.158,0.001),seq(0.16,0.25,0.01))
ms_mcf7 <- NULL
for (i in 1:length(thresh_ms_mcf7)) {
  filt <- mcf7 %>% filter(cv <= thresh_ms_mcf7[i])
  ms_mcf7[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
}

pc3$cv <- pc3$sigma_star/pc3$V1
thresh_pr_pc3 <- seq(0.23,0.4,0.01)
prec_pc3 <- NULL
n_pc3 <- NULL
for (i in 1:length(thresh_pr_pc3)) {
  filt <- pc3 %>% filter(cv <= thresh_pr_pc3[i])
  prec_pc3[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
}

thresh_ms_pc3 <- c(seq(0.134,0.158,0.001),seq(0.16,0.25,0.01))
ms_pc3 <- NULL
for (i in 1:length(thresh_ms_pc3)) {
  filt <- pc3 %>% filter(cv <= thresh_ms_pc3[i])
  ms_pc3[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
}

a375$cv <- a375$sigma_star/a375$V1
thresh_pr_a375 <- seq(0.23,0.4,0.01)
prec_a375 <- NULL
for (i in 1:length(thresh_pr_a375)) {
  filt <- a375 %>% filter(cv <= thresh_pr_a375[i])
  prec_a375[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
}

thresh_ms_a375 <- c(seq(0.13,0.158,0.002),seq(0.16,0.25,0.01))
ms_a375 <- NULL
for (i in 1:length(thresh_ms_a375)) {
  filt <- a375 %>% filter(cv <= thresh_ms_a375[i])
  ms_a375[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
}

vcap$cv <- vcap$sigma_star/vcap$V1
thresh_pr_vcap <- seq(0.23,0.4,0.01)
prec_vcap <- NULL
n_vcap <- NULL
for (i in 1:length(thresh_pr_vcap)) {
  filt <- vcap %>% filter(cv <= thresh_pr_vcap[i])
  prec_vcap[i] <- prec_eval(true = filt$value,pred = filt$V1,thresh = 0.2)
  n_vcap[i] <- length(which(filt$V1<=0.2))
}

thresh_ms_vcap <- c(seq(0.14,0.158,0.001),seq(0.16,0.25,0.01))
ms_vcap <- NULL
for (i in 1:length(thresh_ms_vcap)) {
  filt <- vcap %>% filter(cv <= thresh_ms_vcap[i])
  ms_vcap[i] <- MSE(y_pred = filt$V1,y_true = filt$value)
}
pdf(file="result_fig3c.pdf",width=7,height=6)
png(file="result_fig3c.png",width=7,height=6,units = "in",res = 300)
plot(thresh_ms_pc3,100*ms_pc3/max(ms_pc3,na.rm=T),ylim = c(0,100),"o",xlim = c(0.13,0.25),col = "black",xlab = "CV threshold",
     ylab = "MSE %" ,lwd = 1,lty = 5,pch = 1,cex = 0.2,yaxs="i")
lines(thresh_ms_mcf7,100*ms_mcf7/max(ms_mcf7,na.rm=T),col = "#e41a1c",lwd = 1,type = "o",lty = 5,pch = 2,cex = 0.2)
lines(thresh_ms_a375,100*ms_a375/max(ms_a375,na.rm=T),col = "#4daf4a",lwd = 1, type = "o",lty = 2 , pch = 3,cex = 0.2)
lines(thresh_ms_vcap,100*ms_vcap/max(ms_vcap,na.rm=T),col = "#377eb8",lwd = 1, type = "o",lty = 5,pch = 4,cex = 0.2)
title("C",adj = 0)
legend("topleft", 
       legend = c("PC3", "VCAP", "A375","MCF7"),
       col = c('black', 
               '#377eb8',"#4daf4a","#e41a1c"),pch =c(1,4,3,2),
       pt.cex = 0.5, lwd = 0.8,
       cex = 0.55, 
       text.col = "black", 
       horiz = F )
dev.off()

pdf(file="result_fig3d.pdf",width=7,height=6)
png(file="result_fig3d.png",width=7,height=6,units = "in",res = 300)
plot(thresh_pr_pc3,prec_pc3,ylim = c(0.5,1.01),"o",xlim = c(0.23,0.3),col = "black",xlab = "CV threshold",
     ylab = "Precision" ,lwd = 1,lty = 5,pch = 1,cex = 0.2,yaxs="i")
lines(thresh_pr_mcf7,prec_mcf7,col = "#e41a1c",lwd = 1,type = "o",lty = 5,pch = 2,cex = 0.2)
lines(thresh_pr_a375,prec_a375,col = "#4daf4a",lwd = 1, type = "o",lty = 2 , pch = 3,cex = 0.2)
lines(thresh_pr_vcap,prec_vcap,col = "#377eb8",lwd = 1, type = "o",lty = 5,pch = 4,cex = 0.2)
title("D",adj = 0)
legend("topleft", 
       legend = c("PC3", "VCAP", "A375","MCF7"),
       col = c('black', 
               '#377eb8',"#4daf4a","#e41a1c"),pch =c(1,4,3,2),
       pt.cex = 0.5, lwd = 0.8,
       cex = 0.55, 
       text.col = "black", 
       horiz = F )
dev.off()
