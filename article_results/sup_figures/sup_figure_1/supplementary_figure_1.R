#Load dependencies
library(tidyverse)


############# Barplot of signature quality #############

#Read signatures dataframe
sigs <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/processed_data/cmap_with_RDkits_initial.rds")

#Factorize the unique quality scores and define the cellines wanted to visualize
x_c <- as.factor(c(1,2,3,4,5,6,7,8))
cells <- c('MCF7','VCAP','PC3','A375','A549','HT29','other')

#Create a count matrix
counts <- matrix(nrow=length(cells),ncol=8)
for (i in 1:8){
  for (j in 1:length(cells)){
    if (cells[j]!='other'){
      counts[j,i] <- nrow(sigs %>% filter(quality==i) %>% filter(cell_id==cells[j]) %>% select(quality))
    }else{
      counts[j,i] <- nrow(sigs %>% filter(quality==i) %>% filter(!(cell_id %in% cells[1:length(cells)-1])) %>% select(quality))
    }
  }
}
rownames(counts) <- cells
colnames(counts) <- as.character(x_c)

#Plot the barplot presented in supplementary figure 1
png("supplementary_figure_1.png", width = 12, height = 9,units="in",res=300)
barplot(counts, 
        col=colors()[c(23,89,12,2,4,38,45)] , 
        border="white", 
        space=0.01, 
        font.axis=2, 
        xlab="quality",
        ylab='counts',
        main = 'Quality distribution',
        legend.text = cells)
dev.off()