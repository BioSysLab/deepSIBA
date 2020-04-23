library(tidyverse)

### supl table 2 #####
all <- readRDS("C:/Users/user/Documents/deepSIBA/preprocessing/data_preprocessing/processed_data/cmap_with_RDkits_initial.rds")
go <- readRDS("C:/Users/user/Documents/deepSIBA/article_results/supplementary_table_data/sup_table_2/A375_go.rds")
cell <- "A375"
quality_num <- "1"
plot_significant <- function(all,go,cell,quality_num) {
  df <- all %>% filter(cell_id == cell) %>% 
    filter(quality == quality_num)
  go_nes <- go[[1]]
  go_pval <- go[[2]]
  go_df <- go_nes[,as.character(df$sig_id)]
  go_pval_df <- go_pval[,as.character(df$sig_id)]
  sig <- go_pval_df < 0.05
  sig <- sig+0
  per_compound <- colSums(sig)
  print(mean(per_compound))
}
