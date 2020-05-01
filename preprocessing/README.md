# Preprocessing overview

This folder contains all the necessary R custom functions to process the CMap data and generate the distances between compounds' affected biological processes.

### Functions

- drug_sigs_for_cell.R : adds quality score to each signature and for each drug/cell line combination selects the signature with the highest quality
- get_cmap_signatures.R : retrieves the lvl 5 z-score normalized gene expression data for the specified signature ids.
- go_path_analysis.R : performs GSEA for GO terms related to biological processes.
- go_term_anno.R : annotates the GO terms with gene identifiers.
- kegg_path_analysis.R : performs GSEA for KEGG signaling pathways.
- tf_enrichment.R : performs transcription factor enrichment with Viper and DoRoTHea.
- vectorized_borda_merging.R : merges effect across cell lines for different levels of biological hierarchy.
- distance_scores.R : calculates paiwise distances between compounds' biological effect.
- distance_score_not_pairwise.R : calculates the distance between 2 different lists.
- CV_test_split.R : performs the splitting of the dataset to train/test parts or N fold CV split.

### Example

The preprocessing_example.R script shows the preprocessing pipeline to create a train and test set for the A375 cell line. The input to each functions is explained either in the script or in the function's source code.

### Data_preprocessing

The required data to follow the preprocessing pipeline are available at https://drive.google.com/drive/folders/1BiyzKBcNh7St_pBS9q0Sqpr-1RsO0A8-?usp=sharing.

Additionally, the lvl-5 z-scored transformed gene expression data of the CMap platform have to be downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742.

An explanation to the way the cmap data is organized can be found at https://docs.google.com/document/d/1q2gciWRhVCAAnlvF2iRLuJ7whrGP6QjpsCMq1yWz7dU/edit#.

CMap : Glossary https://clue.io/connectopedia/glossary

### R packages required

- [tidyverse](https://www.tidyverse.org/)
- [doFuture](https://github.com/HenrikBengtsson/doFuture)
- [GeneExpressionSignature](https://www.bioconductor.org/packages/release/bioc/html/GeneExpressionSignature.html)
- [cmapR](https://github.com/cmap/cmapR)
- [rhdf5](https://www.bioconductor.org/packages/release/bioc/html/rhdf5.html)
- [AnnotationDbi](https://bioconductor.org/packages/release/bioc/html/AnnotationDbi.html)
- [org.Hs.eg.db](http://bioconductor.org/packages/release/data/annotation/html/org.Hs.eg.db.html)
- [fgsea](https://bioconductor.org/packages/release/bioc/html/fgsea.html)
- [gage](https://bioconductor.org/packages/release/bioc/html/gage.html)
- [EGSEAdata](http://bioconductor.org/packages/release/data/experiment/html/EGSEAdata.html)
- [topGO](https://bioconductor.org/packages/release/bioc/html/topGO.html)
- [GO.db](http://bioconductor.org/packages/release/data/annotation/html/GO.db.html)
- [CARNIVAL](https://github.com/saezlab/CARNIVAL)
- [viper](https://www.bioconductor.org/packages/release/bioc/html/viper.html)





