from __future__ import division, print_function
import numpy as np
from numpy import inf, ndarray
import pandas as pd
import tensorflow as tf
import os
import random
import keras
import sklearn
import re
from keras import optimizers, losses, regularizers
import keras.backend as K
from keras.models import model_from_json, load_model, Model
from tempfile import TemporaryFile
from keras import layers
from keras.callbacks import History, ReduceLROnPlateau
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Layer
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
from NGF_layers.features import one_of_k_encoding, one_of_k_encoding_unk, atom_features, bond_features, num_atom_features, num_bond_features, padaxis, tensorise_smiles, concat_mol_tensors
from NGF_layers.graph_layers import temporal_padding, neighbour_lookup, NeuralGraphHidden, NeuralGraphOutput
from math import ceil
from sklearn.metrics import precision_score, accuracy_score, mean_squared_error
from utility.gaussian import GaussianLayer, custom_loss
from utility.evaluator import r_square, get_cindex, pearson_r, mse_sliced, model_evaluate
from custom_layers.model_creator import multistage_autoenc
from custom_layers.model_creator import stage_creator, encode_smiles, add_new_layer

#Define siamese encoder
def enc_graph(params,encoder_params):
    ### encode smiles
    atoms0 = Input(name='atom_inputs', shape=(params["max_atoms"], params["num_atom_features"]),dtype = 'float32')
    bonds = Input(name='bond_inputs', shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"]),dtype = 'float32')
    edges = Input(name='edge_inputs', shape=(params["max_atoms"], params["max_degree"]), dtype='int32')

    [model_enc_1, model_dec_pre_act_1, model_dec_after_act_1] = stage_creator(encoder_params,1,conv = True)
    [model_enc_2, model_dec_pre_act_2, model_dec_after_act_2] = stage_creator(encoder_params,2,conv = True)
    [model_enc_3, model_dec_pre_act_3, model_dec_after_act_3] = stage_creator(encoder_params,3,conv = True)

    graph_conv_1 = model_enc_1([atoms0,bonds,edges])
    graph_conv_2 = model_enc_2([graph_conv_1,bonds,edges])
    graph_conv_3 = model_enc_3([graph_conv_2,bonds,edges])

    g4=keras.layers.Conv1D(params["conv1d_filters"], params["conv1d_size"], activation=None, use_bias=False, kernel_initializer='glorot_uniform')(graph_conv_3)
    g4= BatchNormalization(momentum=0.6)(g4)
    g4 = Activation('relu')(g4)
    g4=keras.layers.Dropout(params["dropout_encoder"])(g4)

    #End of encoding
    graph_encoder = keras.Model(inputs=[atoms0, bonds, edges], outputs= g4)

    #print(graph_encoder.summary())
    return graph_encoder

#Define operations of distance module after the siamese encoders
def siamese_model(params,encoder_params):
    atoms0_1 = Input(name='atom_inputs_1', shape=(params["max_atoms"], params["num_atom_features"]),dtype = 'float32')
    bonds_1 = Input(name='bond_inputs_1', shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"]),dtype = 'float32')
    edges_1 = Input(name='edge_inputs_1', shape=(params["max_atoms"], params["max_degree"]), dtype='int32')

    atoms0_2 = Input(name='atom_inputs_2', shape=(params["max_atoms"], params["num_atom_features"]),dtype = 'float32')
    bonds_2 = Input(name='bond_inputs_2', shape=(params["max_atoms"], params["max_degree"], params["num_bond_features"]),dtype = 'float32')
    edges_2 = Input(name='edge_inputs_2', shape=(params["max_atoms"], params["max_degree"]), dtype='int32')

    graph_encoder = enc_graph(params,encoder_params)

    encoded_1 = graph_encoder([atoms0_1,bonds_1,edges_1])
    encoded_2 = graph_encoder([atoms0_2,bonds_2,edges_2])

    product_layer = keras.layers.Lambda(lambda tensors:K.batch_dot(tensors[0],tensors[1],axes=2))
    product = product_layer([encoded_1, encoded_2])

    conv1 = keras.layers.Conv1D(params["conv1d_filters_dist"][0], params["conv1d_size_dist"][0], activation=None, use_bias=False, kernel_initializer='glorot_uniform')(product)
    conv1 = BatchNormalization(momentum=0.6)(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = keras.layers.Dropout(params["dropout_dist"])(conv1)

    conv2 = keras.layers.Conv1D(params["conv1d_filters_dist"][1], params["conv1d_size_dist"][1], activation=None, use_bias=False, kernel_initializer='glorot_uniform')(conv1)
    conv2 = BatchNormalization(momentum=0.6)(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = keras.layers.Dropout(params["dropout_dist"])(conv2)

    conv2_pool = keras.layers.MaxPooling1D(pool_size= params["pool_size"], strides=None, padding='valid', data_format='channels_last')(conv2)
    conv2_pool = BatchNormalization(momentum=0.6)(conv2_pool)
    conv2_pool = keras.layers.Flatten()(conv2_pool)

    fc1 = keras.layers.Dense(params["dense_size"][0],activation = None,kernel_regularizer=regularizers.l2(params["l2reg"]), kernel_initializer='glorot_normal')(conv2_pool)
    fc1 = BatchNormalization(momentum=0.6)(fc1)
    fc1 = Activation('relu')(fc1)
    fc1 = keras.layers.Dropout(params["dropout_dist"])(fc1)


    fc2 = keras.layers.Dense(params["dense_size"][1],activation = None,kernel_regularizer=regularizers.l2(params["l2reg"]), kernel_initializer='glorot_normal')(fc1)
    fc2 = BatchNormalization(momentum=0.6)(fc2)
    fc2 = Activation('relu')(fc2)
    fc2 = keras.layers.Dropout(params["dropout_dist"])(fc2)

    fc3 = keras.layers.Dense(params["dense_size"][2],activation = None,kernel_regularizer=regularizers.l2(params["l2reg"]), kernel_initializer='glorot_normal')(fc2)
    fc3 = BatchNormalization(momentum=0.6)(fc3)
    fc3 = Activation('relu')(fc3)
    fc3 = keras.layers.Dropout(params["dropout_dist"])(fc3)

    #Final Gaussian Layer to predict mean distance and standard deaviation of distance
    mu, sigma = GaussianLayer(1, name='main_output')(fc3)
    siamese_net = Model(inputs = [atoms0_1, bonds_1, edges_1, atoms0_2, bonds_2, edges_2], outputs = mu)

    thresh = params["dist_thresh"] #threshold to consider similars
    adam = keras.optimizers.Adam(lr = params["lr"], beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
    siamese_net.compile(optimizer = adam,loss= custom_loss(sigma),metrics=['mse', get_cindex, r_square, pearson_r, mse_sliced(thresh)])


    #int_net = keras.Model(inputs=[atoms0_1,bonds_1,edges_1,atoms0_2,bonds_2,edges_2],outputs=fc6)
    #print(int_net.summary())
    return siamese_net
