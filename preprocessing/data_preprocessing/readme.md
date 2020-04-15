# Data_preprocessing overview

The data can be found at https://drive.google.com/drive/folders/1BiyzKBcNh7St_pBS9q0Sqpr-1RsO0A8-?usp=sharing.

### cmap

This folder contains the data and information retrieved from the CMap platform. The last file needed to run the examples pipeline is: **GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz** which can be downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742.

### processed_data

Contains the preprocessed dataframe after selecting the highest quality signature for each drug-cell line combination.

### utility

Contains the ECFP4 Tanimoto similarities between the available compounds for each cell line and their rdkit canonical smiles. Additionally a mapping between pert ids and canonical smiles is provided.