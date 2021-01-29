# DeepSIBA: chemical structure-based inference of biological alterations using deep learning
### Christos Fotis<sup>1(+)</sup>, Nikolaos Meimetis<sup>1+</sup>, Antonios Sardis<sup>1</sup>, Leonidas G.Alexopoulos<sup>1,2(*)</sup>
 #### 1. BioSys Lab, National Technical University of Athens, Athens, Greece.
#### 2. ProtATonce Ltd, Athens, Greece.

(+)Equal contributions

(*)Correspondence to: leo@mail.ntua.gr

Github repository of the study:
> DeepSIBA: chemical structure-based inference of biological alterations using deep learning <br>
> C.Fotis<sup>1(+)</sup>, N.Meimetis<sup>1+</sup>, A.Sardis<sup>1</sup>, LG. Alexopoulos<sup>1,2(*)</sup>



## Abstract
![graph_abs_fl_01](https://user-images.githubusercontent.com/48244638/80760167-251cc900-8b41-11ea-9922-b4a1887a181d.png)
Predicting whether a chemical structure leads to a desired or adverse biological effect can have a significant impact for in silico drug discovery. In this study, we developed a deep learning model where compound structures are represented as graphs and then linked to their biological footprint. To make this complex problem computationally tractable, compound differences were mapped to biological effect alterations using Siamese Graph Convolutional Neural Networks. The proposed model was able to encode molecular graph pairs and identify structurally dissimilar compounds that affect similar biological processes with high precision. Additionally, by utilizing deep ensembles to estimate uncertainty, we were able to provide reliable and accurate predictions for chemical structures that are very different from the ones used during training. Finally, we present a novel inference approach, where the trained models are used to estimate the signaling pathway signature of a compound perturbation, using only its chemical structure as input, and subsequently identify which substructures influenced the predicted pathways. As a use case, this approach was used to infer important substructures and affected signaling pathways of FDA-approved anticancer drugs.

## Clone
```bash
# clone the source code on your directory
$ git clone https://github.com/BioSysLab/deepSIBA.git
# For a Tensorflow 2.3 implementation see here: https://github.com/NickMeim/deepSIBA_tf2
# For a simplified Pytorch implmentation see here: https://github.com/NickMeim/deepSIBA_pytorch
```

## Data
For deep learning applications follow the instructions in the data and screening subfolders of the learning folder.
For the general Gene Expression data used and data used for the figures of the article, see instructions in every folder of the article_results and preprocessing sections.
