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
from sklearn.metrics import mean_squared_error
from utility.gaussian import GaussianLayer, custom_loss, ConGaussianLayer
from utility.evaluator import r_square, get_cindex, pearson_r, mse_sliced, model_evaluate
from utility.Generator import train_generator,preds_generator
from dot_siamese_model import enc_graph, siamese_model
from pathlib import Path

def siba_trainer(train_params, model_params):
    get_all = []
    #print("HI")
    if train_params["split"] == "train_test_split":
        outer_loop = train_params["number_folds"]
    elif train_params["split"] == "5_fold_cv_split":
        outer_loop = train_params["number_folds"]
    elif train_params["split"] == "alldata":
        outer_loop = train_params["number_folds"]
    #Load unique smiles and tensorize them
    smiles = pd.read_csv("data/" + train_params["cell_line"] + "/" + train_params["cell_line"] + "q1smiles.csv", index_col=0)
    X_atoms, X_bonds, X_edges = tensorise_smiles(smiles.x, model_params["max_degree"], model_params["max_atoms"])
    smiles=list(smiles['x'])
    for i in outer_loop:
        if train_params["split"] == "train_test_split":
            df = pd.read_csv("data/" + train_params["cell_line"] + "/" + "train_test_split/" + "train.csv",index_col=0).reset_index(drop=True)
            df_cold = pd.read_csv("data/" + train_params["cell_line"] + "/" + "train_test_split/" + "test.csv",index_col=0).reset_index(drop=True)
            smiles_cold = list(set(list(df_cold['rdkit.x'])+list(df_cold['rdkit.y'])))
            X_atoms_cold, X_bonds_cold, X_edges_cold = tensorise_smiles(smiles_cold, max_degree=5, max_atoms = 60)
            if train_params["test_value_norm"]:
                Y_cold = df_cold.value
            else:
                Y_cold = df_cold.value
                Y_cold = Y_cold/2
        elif train_params["split"] == "alldata":
            df = pd.read_csv("data/" + train_params["cell_line"] + "/" + "alldata/" + "alldata_" + train_params["cell_line"] + ".csv",index_col=0).reset_index(drop=True)
            df_cold = pd.read_csv("data/" + train_params["cell_line"] + "/" + "train_test_split/" + "test.csv",index_col=0).reset_index(drop=True)
            smiles_cold = list(set(list(df_cold['rdkit.x'])+list(df_cold['rdkit.y'])))
            X_atoms_cold, X_bonds_cold, X_edges_cold = tensorise_smiles(smiles_cold, max_degree=5, max_atoms = 60)
            if train_params["test_value_norm"]:
                Y_cold = df_cold.value
            else:
                Y_cold = df_cold.value
                Y_cold = Y_cold/2
        elif train_params["split"] == "5_fold_cv_split":
            df = pd.read_csv("data/" + train_params["cell_line"] + "/" + "5_fold_cv_split/" + "fold_%s/train_%s.csv" %(i+1,i+1),index_col=0).reset_index(drop=True)
            df_cold = pd.read_csv("data/" + train_params["cell_line"] + "/" + "5_fold_cv_split/" + "fold_%s/val_%s.csv" %(i+1,i+1),index_col=0).reset_index(drop=True)
            smiles_cold = list(set(list(df_cold['rdkit.x'])+list(df_cold['rdkit.y'])))
            X_atoms_cold, X_bonds_cold, X_edges_cold = tensorise_smiles(smiles_cold, max_degree=5, max_atoms = 60)
            if train_params["test_value_norm"]:
                Y_cold = df_cold.value
            else:
                Y_cold = df_cold.value
                Y_cold = Y_cold/2
        Path(train_params["output_dir"] + "/" + "fold_%s/cold/mu"%i).mkdir(parents=True, exist_ok=True)
        Path(train_params["output_dir"] + "/" + "fold_%s/cold/sigma"%i).mkdir(parents=True, exist_ok=True)
        Path(train_params["output_dir"] + "/" + "fold_%s/models"%i).mkdir(parents=True, exist_ok=True)
        Path(train_params["output_dir"] + "/" + "fold_%s/performance"%i).mkdir(parents=True, exist_ok=True)
        cold_preds_mus = []
        cold_preds_sigmas = []
        n = train_params["nmodel_start"]
        while n < train_params["N_ensemble"]:
            deepsiba = siamese_model(model_params)
            rlr = ReduceLROnPlateau(monitor='loss', factor=0.5,patience=3, min_lr=0.00001, verbose=1, min_delta=1e-5)
            term=keras.callbacks.TerminateOnNaN()
            bs = train_params["batch_size"]
            NUM_EPOCHS = train_params["epochs"]
            df = df.sample(frac=1).reset_index(drop=True)
            NUM_TRAIN = len(df)
            trainGen=train_generator(bs,df,smiles,X_atoms, X_bonds, X_edges)
            check = NUM_EPOCHS-1
            history = deepsiba.fit_generator(trainGen,
                        steps_per_epoch= ceil(NUM_TRAIN/bs),
                        epochs=NUM_EPOCHS,
                        verbose = 1,
                        shuffle = True,
                        callbacks= [term, rlr])
            if history.history["r_square"][len(history.history["r_square"])-1] < 0.7:
                history = deepsiba.fit_generator(trainGen,
                            steps_per_epoch= ceil(NUM_TRAIN/bs),
                            epochs = 10,
                            verbose = 1,
                            shuffle = True,
                            callbacks= [term, rlr])
            if history.history["r_square"][len(history.history["r_square"])-1] >= 0.7:
                deepsiba.save_weights(train_params["output_dir"] + "/" + "fold_%s/models/"%i + "model_%s.h5"%n)
                gaussian = keras.Model(deepsiba.inputs, deepsiba.get_layer('main_output').output)
                pr_steps=ceil(len(df_cold)/train_params["predict_batch_size"])
                PredGen=preds_generator(train_params["predict_batch_size"],df_cold,smiles_cold,X_atoms_cold, X_bonds_cold, X_edges_cold,gaussian)
                y_pred1=[]
                y_pred2=[]
                for g in range(pr_steps):
                    cold_pred=list(next(PredGen))
                    y_pred1=y_pred1+list(cold_pred[0])
                    y_pred2=y_pred2+list(cold_pred[1])
                y_pred1=np.array(y_pred1)
                y_pred2=np.array(y_pred2)
                if (len(y_pred1[np.where(y_pred1 <= train_params["prec_threshold"])])>0):
                    get = model_evaluate(y_pred1,Y_cold,train_params["prec_threshold"],df_cold)
                    get.to_csv(train_params["output_dir"] + "/" + "fold_%s/performance/"%i + "model_%s.csv"%n)
                cold_preds_mus.append(y_pred1)
                np.save(train_params["output_dir"] + "/" + "fold_%s/cold/mu/"%i + "cold_mu_%s.npy"%n, y_pred1)
                cold_preds_sigmas.append(y_pred2)
                np.save(train_params["output_dir"] + "/" + "fold_%s/cold/sigma/"%i + "cold_sigma_%s.npy"%n, y_pred2)
                n = n + 1
        mu_star=np.mean(cold_preds_mus,axis=0)
        sigma_star = np.sqrt(np.mean(cold_preds_sigmas + np.square(cold_preds_mus), axis = 0) - np.square(mu_star))
        cv_star = sigma_star/mu_star
        if (len(mu_star[np.where(mu_star <= train_params["prec_threshold"])])>0):
            get_fold = model_evaluate(mu_star,Y_cold,train_params["prec_threshold"],df_cold)
            get_fold.to_csv(train_params["output_dir"] + "/" + "fold_%s/ensemble_performance.csv"%i)
            get_all.append(get_fold)
        df_cold['mu'] = mu_star
        df_cold['cv'] = cv_star
        df_cold.to_csv(train_params["output_dir"] + "/" + "fold_%s/ensemble_preds_dataframe.csv"%i)
    return(get_all)
