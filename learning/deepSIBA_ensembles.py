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
from pathlib import Path
from utility.Generator import preds_generator

def siba_val_loader(test_params, model_params,deepsiba):
    i=test_params["fold_id"]
    if test_params["split"] == "train_test_split":
        df_cold = pd.read_csv("data/" + test_params["cell_line"] + "/" + "train_test_split/" + "test.csv",index_col=0).reset_index(drop=True)
        smiles_cold = list(set(list(df_cold['rdkit.x'])+list(df_cold['rdkit.y'])))
        X_atoms_cold, X_bonds_cold, X_edges_cold = tensorise_smiles(smiles_cold,  model_params["max_degree"], model_params["max_atoms"])
    elif test_params["split"] == "5_fold_cv_split":
        df_cold = pd.read_csv("data/" + test_params["cell_line"] + "/" + "5_fold_cv_split/" + "fold_%s/val_%s.csv" %(i+1,i+1),index_col=0).reset_index(drop=True)
        smiles_cold = list(set(list(df_cold['rdkit.x'])+list(df_cold['rdkit.y'])))
        X_atoms_cold, X_bonds_cold, X_edges_cold = tensorise_smiles(smiles_cold, model_params["max_degree"], model_params["max_atoms"])
    if not(test_params["test_value_norm"]):
        df_cold.value = df_cold.value/2

    cold_preds_mus = []
    cold_preds_sigmas = []
    n=int(0)
    while n < test_params["N_ensemble"]:
        if test_params["split"] == "train_test_split":
            deepsiba.load_weights("trained_models/"+test_params["cell_line"]+ "/" + test_params["split"]+"/models/" +test_params["name_pattern"] +"_%s.h5"%n)
        elif test_params["split"] == "5_fold_cv_split":
            deepsiba.load_weights("trained_models/" + test_params["cell_line"]+ "/"+test_params["split"]+"/fold_%s/models/"%(i+1) +test_params["name_pattern"] +"_%s.h5"%n)
        gaussian = keras.Model(deepsiba.inputs, deepsiba.get_layer('main_output').output)
        pr_steps=ceil(len(df_cold)/test_params["predict_batch_size"])
        PredGen=preds_generator(test_params["predict_batch_size"],df_cold,smiles_cold,X_atoms_cold, X_bonds_cold, X_edges_cold,gaussian)
        y_pred1=[]
        y_pred2=[]
        for g in range(pr_steps):
            cold_pred=list(next(PredGen))
            y_pred1=y_pred1+list(cold_pred[0])
            y_pred2=y_pred2+list(cold_pred[1])
        y_pred1=np.array(y_pred1)
        y_pred2=np.array(y_pred2)
        cold_preds_mus.append(y_pred1)
        cold_preds_sigmas.append(y_pred2)
        n = n + 1
    mu_star=np.mean(cold_preds_mus,axis=0)
    sigma_star = np.sqrt(np.mean(cold_preds_sigmas + np.square(cold_preds_mus), axis = 0) - np.square(mu_star))
    cv_star = sigma_star/mu_star
    df_cold['mu'] = mu_star
    df_cold['cv'] = cv_star
    return df_cold
