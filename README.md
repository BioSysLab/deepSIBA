# Still under construction!
# DeepSIBA: Chemical Structure-based Inference of Biological Alterations
### Christos Fotis<sup>1(+)</sup>, Nikolaos Meimetis<sup>1+</sup>, Antonios Sardis<sup>1</sup>, Leonidas G.Alexopoulos<sup>1,2(*)</sup>
 #### 1. BioSys Lab, National Technical University of Athens, Athens, Greece.
#### 2. ProtATonce Ltd, Athens, Greece.

(+)Equal contributions

(*)Correspondence to: leo@mail.ntua.gr

Github repository of the study:
> DeepSIBA: Chemical Structure-based Inference of Biological Alterations <br>
> C.Fotis<sup>1(+)</sup>, N.Meimetis<sup>1+</sup>, A.Sardis<sup>1</sup>, LG. Alexopoulos<sup>1,2(*)</sup>



## Abstract
![graph_abs_fl_01](https://user-images.githubusercontent.com/48244638/80760167-251cc900-8b41-11ea-9922-b4a1887a181d.png)
Predicting whether a chemical structure shares a desired biological effect can have a significant impact for in-silico compound screening in early drug discovery.  In this study, we developed a deep learning model where compound structures are represented as graphs and then linked to their biological footprint. To make this complex problem computationally tractable, compound differences were mapped to biological effect alterations using Siamese Graph Convolutional Neural Networks. The proposed model was able to learn new representations from chemical structures and identify structurally dissimilar compounds that affect similar biological processes with high precision. Additionally, by utilizing deep ensembles to estimate uncertainty, we were able to provide more reliable and accurate predictions for chemical structures that are very different from the ones used to the train the models. Finally, we present a novel inference approach, where the trained models are used to provide an estimate of a compound’s effect on signaling pathways, using only its chemical structure.


## Model Requirements
- Install [Tensorflow (version 1.x)](https://www.tensorflow.org/install/gpu)
- Install [Numpy](https://pypi.org/project/numpy/)
- Install [Pandas](https://pandas.pydata.org/)
- Install [rdkit](https://www.rdkit.org/docs/Install.html)

## Main R Packages used in data pre-processing
- [topGO](https://bioconductor.org/packages/release/bioc/html/topGO.html)
- [FGSEA](https://bioconductor.org/packages/release/bioc/html/fgsea.html)
- [Gene Expression Signature](https://www.bioconductor.org/packages/release/bioc/html/GeneExpressionSignature.html)
- [AnnotationDbi](https://bioconductor.org/packages/release/bioc/html/AnnotationDbi.html)
- [tidyverse](https://www.tidyverse.org/)

## Clone
```bash
# clone the source code on your directory
$ git clone https://github.com/BioSysLab/deepSIBA.git
```

## Data
For deep learning applications follow the instructions in the data and screening subfolders of the learning folder.
For the general Gene Expression data used and data used for the figures of the article, see instructions in every folder of the article_results and preprocessing sections.
