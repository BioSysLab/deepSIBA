from __future__ import division, print_function
from comet_ml import Experiment
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import tensorflow as tf
import os
import random
import keras
import sklearn
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
import re
from keras import optimizers
from keras import losses
from keras import regularizers
import keras.backend as K
from keras.models import model_from_json
from keras.models import load_model, Model
from tempfile import TemporaryFile
from keras import layers
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation
from keras.layers import CuDNNLSTM, Dense, Bidirectional, Dropout, Layer
from keras.initializers import glorot_normal
from keras.regularizers import l2
from functools import partial
from multiprocessing import cpu_count, Pool
from keras.utils.generic_utils import Progbar
from copy import deepcopy
from NGF.utils import filter_func_args, mol_shapes_to_dims
import NGF.utils
import NGF_layers.features
import NGF_layers.graph_layers
from NGF_layers.features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, bond_features, num_atom_features, num_bond_features
from NGF_layers.features import padaxis, tensorise_smiles, concat_mol_tensors
from NGF_layers.graph_layers import temporal_padding, neighbour_lookup, NeuralGraphHidden, NeuralGraphOutput
from math import ceil
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from utility.gaussian import GaussianLayer

#Define siamese encoder
def enc_mols(params, lr_value, conv_width, fp_length):
        
    ### encode smiles
    
    atoms0 = Input(name='atom_inputs', shape=(max_atoms, num_atom_features),dtype = 'float32')
    bonds = Input(name='bond_inputs', shape=(max_atoms, max_degree, num_bond_features),dtype = 'float32')
    edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')

    g1 = NeuralGraphHidden(conv_width , activ = None, bias = True , init = 'glorot_normal')([atoms0,bonds,edges])
    g1 = BatchNormalization(momentum=0.6)(g1)
    g1 = Activation('relu')(g1)
    #g1 = keras.layers.Dropout(0.25)(g1) #this enables dropout also in test-time
    g2 = NeuralGraphHidden(conv_width , activ = None, bias = True , init = 'glorot_normal')([g1,bonds,edges])
    g2 = BatchNormalization(momentum=0.6)(g2)
    g2 = Activation('relu')(g2)
    #g2 =keras.layers.Dropout(0.25)(g2)
    g3 = NeuralGraphHidden(conv_width , activ = None, bias = True , init = 'glorot_normal')([g2,bonds,edges])
    g3 = BatchNormalization(momentum=0.6)(g3)
    g3 = Activation('relu')(g3)
    #g3 =keras.layers.Dropout(0.25)(g3)
    
    g4=keras.layers.Conv1D(128, 29, activation=None, use_bias=False, kernel_initializer='glorot_uniform')(g3)
    g4= BatchNormalization(momentum=0.6)(g4)
    g4 = Activation('relu')(g4)
    g4=keras.layers.Dropout(0.25)(g4)
    

    #End of encoding
    interactionModel = keras.Model(inputs=[atoms0, bonds, edges], outputs= g4)

    print(interactionModel.summary())
    return interactionModel

#Define operations of distance module after the siamese encoders
def net(max_atoms, num_atom_features,max_degree, num_bond_features,encoder_mols):
	from utility.gaussian import GaussianLayer
    
	# Initialize model
	atoms0_1 = Input(name='atom_inputs_1', shape=(max_atoms, num_atom_features),dtype = 'float32')
	bonds_1 = Input(name='bond_inputs_1', shape=(max_atoms, max_degree, num_bond_features),dtype = 'float32')
	edges_1 = Input(name='edge_inputs_1', shape=(max_atoms, max_degree), dtype='int32')

	atoms0_2 = Input(name='atom_inputs_2', shape=(max_atoms, num_atom_features),dtype = 'float32')
	bonds_2 = Input(name='bond_inputs_2', shape=(max_atoms, max_degree, num_bond_features),dtype = 'float32')
	edges_2 = Input(name='edge_inputs_2', shape=(max_atoms, max_degree), dtype='int32')

	encoded_1 = encoder_mols([atoms0_1,bonds_1,edges_1])
	encoded_2 = encoder_mols([atoms0_2,bonds_2,edges_2])

	L1_layer = keras.layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
	L1_distance = L1_layer([encoded_1, encoded_2])

	fc1=keras.layers.Conv1D(128, 17, activation=None, use_bias=False, kernel_initializer='glorot_uniform')(L1_distance)
	fc1= BatchNormalization(momentum=0.6)(fc1)
	fc1= Activation('relu')(fc1)
	fc1=keras.layers.Dropout(0.25)(fc1)

	fc2=keras.layers.Conv1D(128, 1, activation=None, use_bias=False, kernel_initializer='glorot_uniform')(fc1)
	fc2= BatchNormalization(momentum=0.6)(fc2)
	fc2= Activation('relu')(fc2)
	fc2=keras.layers.Dropout(0.25)(fc2)


	fc3=keras.layers.MaxPooling1D(pool_size= 4, strides=None, padding='valid', data_format='channels_last')(fc2)
	fc3 = BatchNormalization(momentum=0.6)(fc3)
	fc3=keras.layers.Flatten()(fc3)

	fc4 = keras.layers.Dense(256,activation = None,kernel_regularizer=regularizers.l2(p['l2reg']), kernel_initializer='glorot_normal')(fc3)
	fc4 = BatchNormalization(momentum=0.6)(fc4)
	fc4 = Activation('relu')(fc4)
	fc4 = keras.layers.Dropout(0.25)(fc4)


	fc5 = keras.layers.Dense(128,activation = None,kernel_regularizer=regularizers.l2(p['l2reg']), kernel_initializer='glorot_normal')(fc4)
	fc5 = BatchNormalization(momentum=0.6)(fc5)
	fc5 = Activation('relu')(fc5)
	fc5 = keras.layers.Dropout(0.3)(fc5)

	fc6 = keras.layers.Dense(128,activation = None,kernel_regularizer=regularizers.l2(p['l2reg']), kernel_initializer='glorot_normal')(fc5)
	fc6 = BatchNormalization(momentum=0.6)(fc6)
	fc6 = Activation('relu')(fc6)
	fc6 = keras.layers.Dropout(0.3)(fc6)


	mu, sigma = GaussianLayer(1, name='main_output')(fc6)


	siamese_net = Model(inputs=[atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2],outputs=mu)
	print(siamese_net.summary())
	return(siamese_net)