# Trained models directory overview

Due to their size, the files containing the weights of trained deepSIBA ensemble models can be downloaded from https://drive.google.com/drive/folders/1TQOXCLV1Z8Lbfbo2BH5e7L48tOxJxWmO?usp=sharing.

The trained models folder is organized into categories depending on the cellular model,

- a375 folder contains the trained models' weights for the A375 cell line, a malignant melanoma cell line.
- pc3 folder contains the trained models' weights for the PC3 cell line, a prostate cancer cell line.
- vcap folder contains the trained models' weights for the VCAP cell line, a prostate cancer cell line.
- mcf7 folder contains the trained models' weights for the MCF7 cell line, a breast cancer cell line.
- merged folder contains the trained models' weights for the merged approach, in which the biological effect of compound perturbations is merged across cellular models, at different levels of biological hierarchy.

Inside each folder, the weight files are organized further into folders based on the selected split.

- train_test, where all distances involving the test compounds are removed from the training set. For the train test split, trained ensembles of 50 models are provided.
- 5_fold_cv_split, 5 fold cross validation split, in which the validation compounds are removed from the appropriate training set. For the 5 fold CV split, trained ensembles of 10 models are provided for each split.
- alldata, contains the weights of models trained on the entirety of each data set. In this case, trained ensembles of 50 models are provided.

