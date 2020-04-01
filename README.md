# DeepSIBA: Chemical Structure-based Inference of Biological Alterations
### Christos Fotis<sup>1(+)</sup>, Nikolaos Meimetis<sup>1+</sup>, Antonios Sardis<sup>1</sup>, Leonidas G.Alexopoulos<sup>1,2(*)</sup>
 #### 1. BioSys Lab, National Technical University of Athens, Athens, Greece.
#### 2. ProtATonce Ltd, Athens, Greece.

(+)Equal contributions

(*)Correspondence to: leo@mail.ntua.gr

Implementation of the paper:
> DeepSIBA: Chemical Structure-based Inference of Biological Alterations <br>
> C.Fotis<sup>1(+)</sup>, N.Meimetis<sup>1+</sup>, A.Sardis<sup>1</sup>, LG. Alexopoulos<sup>1,2(*)</sup>



## Abstract
![figure1_01_fl](https://user-images.githubusercontent.com/48244638/78174099-66965780-7461-11ea-84a4-2003cdcd51cc.png)
Predicting whether a chemical structure shares a desired biological effect can have a significant impact for in-silico compound screening in early drug discovery.  In this study, we developed a deep learning model where compound structures are represented as graphs and then linked to their biological footprint. To make this complex problem computationally tractable, compound differences were mapped to biological effect alterations using Siamese Graph Convolutional Neural Networks. The proposed model was able to learn new representations from chemical structures and identify structurally dissimilar compounds that affect similar biological processes with high precision. Additionally, by utilizing deep ensembles to estimate uncertainty, we were able to provide more reliable and accurate predictions for chemical structures that are very different from the ones used to the train the models. Finally, we present a novel inference approach, where the trained models are used to provide an estimate of a compound’s effect on signaling pathways, using only its chemical structure.


## Model Requirements
- Install [Tensorflow 1.15](https://www.tensorflow.org/install/gpu)
- Install [Numpy 1.61.1](https://pypi.org/project/numpy/)
- Python version >= 3.4.3 is required

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
Here you can find all the data produced and used in this study.
The data and models folders must replace the corresponding empty folders existing on Github, in order to recreate the study. 
If there are already data and files in these folders for another study, just the contents of the downloaded folder must be copied in the data folder.
 
1.[data](http://google.com)

2.[models](http://google.com)
