# Data directory overview

Due to their size, the data files can be downloaded from https://drive.google.com/drive/folders/1qjmec4-DHukr57tB_5y9AiVJj9KS0J7s?usp=sharing.

The data folders contain the required data to train and test the deepSIBA models. The data are split into categories depending on the cellular model,

- a375 folder contains the data for the A375 cell line, a malignant melanoma cell line.
- pc3 folder contains the data for the PC3 cell line, a prostate cancer cell line.
- vcap folder contains the data for the VCAP cell line, a prostate cancer cell line.
- mcf7 folder contains the data for the MCF7 cell line, a breast cancer cell line.
- merged folder contains the data for the merged approach, in which the biological effect of compound perturbations is merged across cellular models, at different levels of biological hierarchy.

Inside each data folder, the data are organized further into folders based on the selected split.

- train_test, where all distances involving the test compounds are removed from the training set.
- 5_fold_cv_split, 5 fold cross validation split, in which the validation compounds are removed from the appropriate training set.
- alldata, contains the data for each cell line without any split.

The files **'cell'**q1smiles.csv contain the rdkit canonical smiles of compounds with available gene expression data for each cell line.