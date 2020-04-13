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
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from deepSIBA_model import siamese_model

def siba_screening(params,model_params):
    query_mol = Chem.MolFromSmiles(params["query_smile"])
    query = Chem.MolToSmiles(query_mol)
    atom_num = query_mol.GetNumAtoms()
    if atom_num > params["atom_limit"]:
        print("The query molecule has more than" + "%s"%params["atom_limit"] + "atoms, the selected models are trained for molecules up to 60 atoms, process terminated!")
    elif atom_num <= params["atom_limit"]:
        smiles = pd.read_csv("data/" + params["cell_line"] + "/" + params["cell_line"] + "q1smiles.csv", index_col=0)
        mol_train = [Chem.MolFromSmiles(smi) for smi in smiles["x"]]
        similarity=np.zeros(shape=(len(mol_train),1),dtype=np.float32)
        fps_ecfp4_train = [AllChem.GetMorganFingerprintAsBitVect(x,2,nBits=1024) for x in mol_train]
        fps_ecfp4_query = AllChem.GetMorganFingerprintAsBitVect(query_mol,2,nBits=1024)
        Path(params["output_dir"]).mkdir(parents=True, exist_ok=True)
        for i in range(len(mol_train)):
            similarities_ecfp4=DataStructs.FingerprintSimilarity(fps_ecfp4_train[i],fps_ecfp4_query)
            similarity[i][0] = similarities_ecfp4
        if np.any(similarity > 0.9):
            print("The selected chemical structure has at least one structural analogue in the training set, chembl and cmap will be screened")
            # screen chembl
            for i in range(9):
                df_screening = pd.read_csv('screening/chembl/part%sinput.csv'%(i+1),index_col=0).reset_index(drop=True)
                df_screening["query"] = query
                smiles_chembl = df_screening['can_smiles']
                smiles_query = df_screening['query']
                X_atoms_cold_1, X_bonds_cold_1, X_edges_cold_1 = tensorise_smiles(smiles_chembl, max_degree=5, max_atoms = params["atom_limit"])
                X_atoms_cold_2, X_bonds_cold_2, X_edges_cold_2 = tensorise_smiles(smiles_query, max_degree=5, max_atoms = params["atom_limit"])
                cold_preds_mus=[]
                cold_preds_sigmas=[]
                siamese_net = siamese_model(model_params)
                for n in range(params["N_models"]):
                    if params["split"] == "train_test_split":
                        siamese_net.load_weights("trained_models/"+params["cell_line"]+ "/" + params["split"]+"/models/" +params["name_pattern"] +"_%s.h5"%n)
                    elif params["split"] == "5_fold_cv_split":
                        siamese_net.load_weights("trained_models/" + params["cell_line"]+ "/"+params["split"]+"/fold_%s/models/"%(params["fold_id"]+1) +params["name_pattern"] +"_%s.h5"%n)
                    elif params["split"] == "alldata":
                        siamese_net.load_weights("trained_models/"+params["cell_line"]+ "/" + params["split"]+"/models/" +params["name_pattern"] +"_%s.h5"%n)
                    elif params["split"] == "custom":
                        siamese_net.load_weights(params["model_path"] + params["name_pattern"] +"_%s.h5"%n)
                    gaussian = keras.Model(siamese_net.inputs, siamese_net.get_layer('main_output').output)
                    cold_pred = gaussian.predict([X_atoms_cold_1,X_bonds_cold_1,X_edges_cold_1,X_atoms_cold_2, X_bonds_cold_2, X_edges_cold_2],batch_size=4096)
                    cold_preds_mus.append(cold_pred[0])
                    cold_preds_sigmas.append(cold_pred[1])
                print('Finished screening for part %s'%(i+1) + '/9 for the Chembl database')
                mu_star=np.mean(cold_preds_mus,axis=0)
                sigma_star = np.sqrt(np.mean(cold_preds_sigmas + np.square(cold_preds_mus), axis = 0) - np.square(mu_star))
                cv_star = sigma_star/mu_star
                df_screening['mu'] = mu_star
                df_screening['cv'] = cv_star
                df_screening.drop(df_screening[df_screening['mu']>params["screening_threshold"]].index, inplace=True)
                df_screening.to_csv(params["output_dir"] + "/results_part%s.csv"%i)
            # screen cmap
            df_screening = pd.read_csv("screening/cmap/cmap_smiles.csv",index_col=0).reset_index(drop=True)
            df_screening["query"] = query
            smiles_cmap = df_screening['x']
            smiles_query = df_screening['query']
            X_atoms_cold_1, X_bonds_cold_1, X_edges_cold_1 = tensorise_smiles(smiles_cmap, max_degree=5, max_atoms = params["atom_limit"])
            X_atoms_cold_2, X_bonds_cold_2, X_edges_cold_2 = tensorise_smiles(smiles_query, max_degree=5, max_atoms = params["atom_limit"])
            cold_preds_mus=[]
            cold_preds_sigmas=[]
            siamese_net = siamese_model(model_params)
            for n in range(params["N_models"]):
                if params["split"] == "train_test_split":
                    siamese_net.load_weights("trained_models/"+params["cell_line"]+ "/" + params["split"]+"/models/" +params["name_pattern"] +"_%s.h5"%n)
                elif params["split"] == "5_fold_cv_split":
                    siamese_net.load_weights("trained_models/" + params["cell_line"]+ "/"+params["split"]+"/fold_%s/models/"%(params["fold_id"]+1) +params["name_pattern"] +"_%s.h5"%n)
                elif params["split"] == "alldata":
                    siamese_net.load_weights("trained_models/"+params["cell_line"]+ "/" + params["split"]+"/models/" +params["name_pattern"] +"_%s.h5"%n)
                elif params["split"] == "custom":
                    siamese_net.load_weights(params["model_path"] + params["name_pattern"] +"_%s.h5"%n)
                gaussian = keras.Model(siamese_net.inputs, siamese_net.get_layer('main_output').output)
                cold_pred = gaussian.predict([X_atoms_cold_1,X_bonds_cold_1,X_edges_cold_1,X_atoms_cold_2, X_bonds_cold_2, X_edges_cold_2],batch_size=4096)
                cold_preds_mus.append(cold_pred[0])
                cold_preds_sigmas.append(cold_pred[1])
            print('Finished screening for CMap database')
            mu_star=np.mean(cold_preds_mus,axis=0)
            sigma_star = np.sqrt(np.mean(cold_preds_sigmas + np.square(cold_preds_mus), axis = 0) - np.square(mu_star))
            cv_star = sigma_star/mu_star
            df_screening['mu'] = mu_star
            df_screening['cv'] = cv_star
            df_screening.drop(df_screening[df_screening['mu']>params["screening_threshold"]].index, inplace=True)
            df_screening.to_csv(params["output_dir"] + "/results_cmap.csv")
        else:
            print("The query chemical structure has no structural analogues in the training molecules, the appropriate training set will be screened")
            df_screening = pd.read_csv("data/" + params["cell_line"] + "/" + params["cell_line"] + "q1smiles.csv", index_col=0)
            df_screening["query"] = query
            smiles_cmap = df_screening['x']
            smiles_query = df_screening['query']
            X_atoms_cold_1, X_bonds_cold_1, X_edges_cold_1 = tensorise_smiles(smiles_cmap, max_degree=5, max_atoms = params["atom_limit"])
            X_atoms_cold_2, X_bonds_cold_2, X_edges_cold_2 = tensorise_smiles(smiles_query, max_degree=5, max_atoms = params["atom_limit"])
            cold_preds_mus=[]
            cold_preds_sigmas=[]
            siamese_net = siamese_model(model_params)
            for n in range(params["N_models"]):
                if params["split"] == "train_test_split":
                    siamese_net.load_weights("trained_models/"+params["cell_line"]+ "/" + params["split"]+"/models/" +params["name_pattern"] +"_%s.h5"%n)
                elif params["split"] == "5_fold_cv_split":
                    siamese_net.load_weights("trained_models/" + params["cell_line"]+ "/"+params["split"]+"/fold_%s/models/"%(params["fold_id"]+1) +params["name_pattern"] +"_%s.h5"%n)
                elif params["split"] == "alldata":
                    siamese_net.load_weights("trained_models/"+params["cell_line"]+ "/" + params["split"]+"/models/" +params["name_pattern"] +"_%s.h5"%n)
                elif params["split"] == "custom":
                    siamese_net.load_weights(params["model_path"] + params["name_pattern"] +"_%s.h5"%n)
                gaussian = keras.Model(siamese_net.inputs, siamese_net.get_layer('main_output').output)
                cold_pred = gaussian.predict([X_atoms_cold_1,X_bonds_cold_1,X_edges_cold_1,X_atoms_cold_2, X_bonds_cold_2, X_edges_cold_2],batch_size=4096)
                cold_preds_mus.append(cold_pred[0])
                cold_preds_sigmas.append(cold_pred[1])
            print('Finished screening the training set for the unknown query molecule')
            mu_star=np.mean(cold_preds_mus,axis=0)
            sigma_star = np.sqrt(np.mean(cold_preds_sigmas + np.square(cold_preds_mus), axis = 0) - np.square(mu_star))
            cv_star = sigma_star/mu_star
            df_screening['mu'] = mu_star
            df_screening['cv'] = cv_star
            df_screening.drop(df_screening[df_screening['mu']>params["screening_threshold"]].index, inplace=True)
            df_screening.to_csv(params["output_dir"] + "/results_trainingset.csv")
