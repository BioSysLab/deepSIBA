library(tidyverse)
library(topGO)
library(org.Hs.eg.db)
library(GO.db)

ds_path <- "C:/Users/user/Documents/phd/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"

### read landmark genes

landmark <- read_tsv(file = "myc_cmap_pathways/cmap_landmark_genes.txt")

### go terms

genes <- factor(x = rep(1,978),levels = c(0,1))
names(genes) <- landmark$`Entrez ID`

GOobject <- new("topGOdata",ontology = "BP", allGenes = genes, annot=annFUN.org, mapping="org.Hs.eg.db", 
                ID = "entrez", nodeSize = 10)

term.genes <- genesInTerm(GOobject, GOobject@graph@nodes)

saveRDS(term.genes,"data/GOterms/goterm_annotation.rds")
