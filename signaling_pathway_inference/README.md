### This directory holds the scripts to perform the signaling pathway inference for a new compound structure.

The additional data required to perform the pathway inference can be downloaded from https://drive.google.com/drive/folders/19i0zn6YDjOqSjBb-wCcz3PRwRaU0XCKp?usp=sharing.

The pathway inference is performed in 2 steps:

- Given a compound's chemical structure and a selected cell line, the appropriate ensemble models are loaded and the distances between the query structure and all training compounds are predicted. Then given a threshold, the training neighbors of the unknown structure are selected. This step is described in PI_step_1.ipynb. 
- From the identified neighbors, their upregulated and downregulated pathways are retrieved. From the neighbors' lists a pathway list is inferred for the query chemical structure. This is performed by PI_step_2.R