# Learning directory overview

This directory contains the required data and source code to implement deepSIBA. 

- The data, trained_models and screening folders contain a readme with the appropriate download instructions and explanations.

The NGF and NGF layers folders contain the source code to implement the graph convolution layers and the appropriate featurization. The code was adapted to Keras from https://github.com/keiserlab/keras-neural-graph-fingerprint.

The utility folder contains the following functions:

- A Keras training generator
- A function to evaluate the performance of deepSIBA
- Custom layer and loss function to implement the Gaussian regression layer.



**The notebook deepSIBA_examples describes how to implement and utilize deepSIBA.**

