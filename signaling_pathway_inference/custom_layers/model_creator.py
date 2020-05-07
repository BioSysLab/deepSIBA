from __future__ import division, absolute_import
import sys
sys.path.append('..')
from functools import partial
from math import ceil
import pandas as pd
import numpy as np
from numpy.random import seed
import tensorflow as tf
from tensorflow import set_random_seed
import keras.backend as K
from keras.layers import Input, BatchNormalization, Activation, add, Lambda, Layer, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from keras.losses import mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from custom_layers.tied_graph_autoencoder import TiedGraphAutoencoder, TiedGraphAutoencoderFP, neighbour_lookup, create_vni_vxi
from utils.data_gen import mask_atoms_by_degree
from NGF.preprocessing import tensorise_smiles
from sklearn.model_selection import train_test_split, StratifiedKFold
from skopt.utils import use_named_args

""" csv_path = '../datasets/cmap_canonical_smiles.csv'
smiles = pd.read_csv(csv_path, index_col=0)
smiles = smiles.loc[(smiles['Atoms'] >= 9) & (smiles['Atoms'] <= 62) & (smiles['0'] != 666)]
print(smiles)
smiles_x = np.array(smiles['0'])

# Specify model callbacks on training
es = EarlyStopping(monitor='val_loss',patience=5, min_delta=0)
rlr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=3, verbose=1, min_lr=0.0000001)

model_params = {
        "num_layers" : 1,
        "max_atoms" : 62,
        "num_atom_features" : 62,
        "num_atom_features_original" : 62,
        "num_bond_features" : 6,
        "max_degree" : 5,
        "conv_width" : [84],
        "fp_length" : [512],
        "activ_enc" : "selu",
        "activ_dec" : "selu",
        "learning_rates" : [0.0078850884983896],
        "learning_rates_fp": [0.0065],
        "losses_conv" : {
                    "neighbor_output": "mean_squared_error",
                    "self_output": "mean_squared_error",
                    },
        "lossWeights" : {"neighbor_output": 1.5, "self_output": 1.0},
        "metrics" : "mse",
        "loss_fp" : "mean_squared_error",
        "enc_layer_names" : ["enc_1", "enc_2", "enc_3"],
        'callbacks' : [es,rlr],
        'adam_decay': 0.0005329142291371636,
        'beta': 5,
        'p': 0.004465204118126482
        }

train_params = {
        "epochs": 1,
        "batch_size" : 256,
        "validation_split" : 0.15
} """

def encode_smiles(max_atoms, num_atom_features, max_degree, num_bond_features):
    atoms = Input(name='atoms', shape=(max_atoms, num_atom_features))
    bonds = Input(name='bonds', shape=(max_atoms, max_degree, num_bond_features))
    edges = Input(name='edge_inputs', shape=(max_atoms, max_degree), dtype='int32')
    return atoms, bonds, edges

def stage_creator(model_params, layer, conv=True):
    """
    Returns a set of stage I or II encoders and decoders as well as the appropriate datasets for training.
    Inputs:
            params: a list of parameters for the models that include max_atoms, number of atom and bond features seperately,
                    max_degree, conv_width, fp_length, activ_enc, activ_dec, optimizer, losses(dict), lossWeights(dict)
                    and metrics
            layer: the layer for which we are creating the autoencoder
            conv: choice between graph convolution(True) or graph fingerprint(False)

    Output: model_dec
        where:
            model_dec: the decoder part of the model which also includes the model for the encoder (can be shown in summary)
    """
    params = model_params.copy()
    layer = layer - 1
    atoms, bonds, edges = encode_smiles(params["max_atoms"],
                        params["num_atom_features"],
                        params["max_degree"],
                        params["num_bond_features"])

    if conv:
        assert params["conv_width"] is not None
        print(f"LAYER {layer}")
        if layer > 0:
            atoms = Input(name='atom_feature_inputs', shape=(params["max_atoms"], params['conv_width'][layer-1]))
            params['num_atom_features'] = params['conv_width'][layer-1]
        # Stage I model
        vxip1 = TiedGraphAutoencoder(params["conv_width"][layer],
                                    original_atom_bond_features= None,
                                    activ=None,
                                    bias=True,
                                    init='glorot_normal',
                                    encode_only=True,
                                    activity_reg = partial(sparse_reg, p=params['p'], beta=params['beta']))([atoms, bonds, edges])
                                    #partial(sparse_reg, p=params['p'], beta=params['beta'])
        vxip1 = BatchNormalization(momentum=0.6)(vxip1)
        #if layer > 0:
        vxip1 = LeakyReLU(alpha=0.3, name='vxi_plus_one')(vxip1)
        #else:
            #vxip1 = Activation('selu', name='vxi_plus_one')(vxip1)
        model_enc = Model(inputs=[atoms, bonds, edges], outputs=[vxip1], name="graph_conv_encoder")
        model_enc.name = params["enc_layer_names"][layer]

        [vni_dot, vxi_dot] = TiedGraphAutoencoder(params["conv_width"][layer],
                                                original_atom_bond_features=(params["num_atom_features"]+params["num_bond_features"]),
                                                activ=None,
                                                bias=True,
                                                init='glorot_normal',
                                                decode_only=True,
                                                tied_to=model_enc.layers[3])([model_enc([atoms, bonds, edges]), bonds, edges])
        model_dec_pre_act = Model(inputs=[atoms, bonds, edges], outputs=[vni_dot, vxi_dot])
        vni_dot = BatchNormalization(momentum=0.6)(vni_dot)
        #if layer > 0:
        vni_dot = LeakyReLU(alpha=0.3, name="neighbor_output")(vni_dot)
        #else:
        #    vni_dot = Activation('selu', name="neighbor_output")(vni_dot)
        vxi_dot = BatchNormalization(momentum=0.6)(vxi_dot)
        vxi_dot = Activation('selu', name="self_output")(vxi_dot)
        model_dec_after_act = Model(inputs=[atoms, bonds, edges], outputs=[vni_dot, vxi_dot])

    else:
        assert params["fp_length"] is not None
        # Stage II model
        vxip1 = Input(name='vxip1', shape=(params["max_atoms"], params["conv_width"][layer]))
        fp_out = TiedGraphAutoencoderFP(params["fp_length"][layer],
                                        activ=None,
                                        bias=True,
                                        init='glorot_normal',
                                        encode=True,
                                        original_atom_bond_features=(params["conv_width"][layer]+ params["num_bond_features"]))([vxip1, bonds, edges])
        fp_out = BatchNormalization(momentum=0.6)(fp_out)
        fp_out = Activation('softmax')(fp_out)
        model_enc = Model(inputs=[vxip1, bonds, edges], outputs=[fp_out], name='encoder_fp')
        model_enc.name = params["enc_layer_names"][layer] + "_fp"

        vxi_dot_fp = TiedGraphAutoencoderFP(params["fp_length"][layer],
                                            activ=None,
                                            bias=True,
                                            init='lecun_normal',
                                            decode=True,
                                            original_atom_bond_features=(params["conv_width"][layer] + params["num_bond_features"]),
                                            tied_to=model_enc.layers[3])([model_enc([vxip1, bonds, edges]), bonds, edges])
        vxi_dot_fp = BatchNormalization(momentum=0.6)(vxi_dot_fp)
        vxi_dot_fp = Activation('selu')(vxi_dot_fp)
        model_dec = Model(inputs=[vxip1, bonds, edges], outputs=vxi_dot_fp)

    if conv:
        return model_enc, model_dec_pre_act, model_dec_after_act
    else:
        return model_dec, model_enc

def untrainable(layer):
    assert isinstance(layer, Layer)
    layer.trainable = False
    return layer

def custom_mse(y_true, y_pred, val_loss):
    mse = mean_squared_error(y_true, y_pred)

    return mse + val_loss

def accumulative_loss(stage_I_val_loss):
    def original(y_true, y_pred):
        return custom_mse(y_true, y_pred, stage_I_val_loss)
    return original

def add_new_layer(model_enc_old, params, train_params, layer, X):
    """
    Adds a new TiedAutoencoder layer instance to the model and sets every other layer as non_trainable in order to train only the
    new one. Used for greedy-layer wise autoencoder training.
    Inputs:
        model_old: the existing Model of the autoencoder so far
        new_layer: the layer which we want to add to the autoencoder, must have the same structure as the old one
                TiedAutoencoderEnc
                BatchNorm
                Activation --> this structure defines the model_enc Model
                --(new layer is inserted here with the same structure as the previous)--
                TiedAutoencoderDec (tied to the first)
                BatchNorm
                Activation --> with the corresponding outputs of the model
        params: the model_params dict
        train_params: the model training parameters
        layer: the current layer number
    Outputs:
        a new model with updated layers
        the encoder part of the new model for the next layer training
    """
    X_atoms, X_bonds, X_edges = X

    atoms, bonds, edges = encode_smiles(params["max_atoms"],
                        params["num_atom_features_original"],
                        params["max_degree"],
                        params["num_bond_features"])

    # For a start, make every other layer non trainable
    model_enc_old.name = "stage_I_encoder_layer_" + str(layer-1)
    model_enc_old.load_weights('layer_{}_stage_I_enc_weights.h5'.format(layer-1))
    model_enc_old.trainable = False

    #Create a new encoder for the next stage
    new_enc, new_dec, _ = stage_creator(params, layer, conv=True)
    new_enc.name = 'stage_I_encoder_layer_' + str(layer)
    new_dec.name = 'stage_I_autoencoder_layer_' + str(layer)


    vxip1 = model_enc_old([atoms, bonds, edges])
    vxip1_new = new_enc([vxip1, bonds, edges])

    create_vni_vxi_part = partial(create_vni_vxi, bonds=bonds, edges=edges)
    vni, vxi = Lambda(create_vni_vxi_part)(vxip1)

    vni_dot = new_dec([vxip1, bonds, edges])[0]
    vxi_dot = new_dec([vxip1, bonds, edges])[1]

    vni_dot = BatchNormalization(momentum=0.6)(vni_dot)
    vni_dot = Activation(params["activ_dec"], name="neighbor_output")(vni_dot)
    vxi_dot = BatchNormalization(momentum=0.6)(vxi_dot)
    vxi_dot = Activation(params["activ_dec"], name="self_output")(vxi_dot)

    enc = Model(inputs = [atoms, bonds, edges], outputs=[vxip1_new])

    opt = Adam(lr=params["learning_rates"][layer-1], beta_1=0.9, beta_2=0.999, epsilon=1e-8,
               decay=params['adam_decay'], amsgrad=False)

    new = Model(inputs=[atoms, bonds, edges], outputs = [vni_dot, vxi_dot])

    new.compile(optimizer=opt,metrics=['mse'], loss=['mse', 'mse'], target_tensors=[vni, vxi])

    new.fit(x=[X_atoms, X_bonds, X_edges],
                            epochs=train_params['epochs'],
                            batch_size=train_params['batch_size'],
                            validation_split=train_params['validation_split'],
                            callbacks=params['callbacks'],
                            verbose=1)

    # Set the weights for the next encoder equal to the trained ones from the new autoencoder encoder part
    # Then, save the new encoders weights into an h5 file for later use
    enc_weights = new.layers[-5].layers[3].get_weights()
    enc.layers[-1].set_weights(enc_weights)
    enc.save_weights('layer_{}_stage_I_enc_weights.h5'.format(layer))
    new.layers[-5].layers[3].save_weights('layer_{}_stage_I_enc_weights_true.h5'.format(layer))
    #print(enc.summary())
    return new, enc

def add_new_layer_fp(model_enc_old, params, train_params, layer, X):
    pass

def multistage_autoenc(smiles_x, num_layers, params, train_params):
    # Create empty lists for outputs
    #val_losses = []
    # X_atom, X_bond, X_edge = tensorise_smiles(smiles_x[:2], max_degree=5, max_atoms=60)
    print('Processing SMILES...')
    #X, val = train_test_split(smiles_x, test_size=train_params["validation_split"], shuffle=True,
                                  #random_state = np.random.randint(1, 10000))
    X_atoms, X_bonds, X_edges = tensorise_smiles(smiles_x, max_degree=5, max_atoms=params['max_atoms'])
    #X_atoms_val, X_bonds_val, X_edges_val = tensorise_smiles(val, max_degree=5, max_atoms=params['max_atoms'])

    vni, vxi = vni_vxi(X_atoms, X_bonds, X_edges)
    #vni_val, vxi_val = vni_vxi(X_atoms_val, X_bonds_val, X_edges_val)
    # Iterate for every layer
    for layer in range(1, num_layers+1):
        opt = Adam(lr=params["learning_rates"][layer-1], beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                    decay=params['adam_decay'], amsgrad=False)
            #########################################################################
            ######################### STAGE I #######################################
            #########################################################################

            #gen = GraphDataGen(X, train_params['batch_size'], params, shuffle=False)
            #valid = GraphDataGen(val, train_params['batch_size'], params, shuffle=False

        if layer == 1:
            stage_I_enc, _, stage_I_dec = stage_creator(params, layer, conv=True)
            stage_I_dec.compile(optimizer=opt, loss=params['losses_conv'], metrics=['mse'])
            stage_I_dec.fit(x=[X_atoms, X_bonds, X_edges], y=[vni, vxi], epochs=train_params['epochs'],validation_split=0.1,
                                callbacks=params['callbacks'],
                                batch_size=train_params['batch_size'])
            stage_I_enc = stage_I_dec.layers[3]
            stage_I_enc.save_weights('layer_{}_stage_I_enc_weights.h5'.format(layer))

            #val_losses.append(stage_I_dec.evaluate(x=[X_atoms_val, X_bonds_val, X_edges_val], y=[vni_val, vxi_val])[0])
        else:
            stage_I_dec, stage_I_enc = add_new_layer(stage_I_enc, params, train_params, layer, X=[X_atoms, X_bonds, X_edges])

        #########################################################################
        ######################### STAGE II ######################################
        #########################################################################
        stage_I_encodings = stage_I_enc.predict([X_atoms, X_bonds, X_edges])
        _, vxi_II = vni_vxi(stage_I_encodings, X_bonds, X_edges)

        stage_II_dec, stage_II_enc = stage_creator(params, layer, conv=False)
        opt = Adam(lr=params["learning_rates_fp"][layer-1], beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                    decay=params['adam_decay'], amsgrad=False)

        stage_II_dec.compile(optimizer=opt, loss=params['loss_fp'], metrics=['mse'])
        stage_II_dec.fit([stage_I_encodings, X_bonds, X_edges], y=[vxi_II],
                            epochs=train_params['epochs'],
                            validation_split=train_params['validation_split'],
                            callbacks=params['callbacks'],
                            batch_size=train_params['batch_size'],
                            verbose=1)
        stage_II_enc.save_weights(f'layer_{layer}_stage_II_enc_weights.h5')

        #stage_I_encodings_val = stage_I_enc.predict([X_atoms_val, X_bonds_val, X_edges_val])
        #_, vxi_val = vni_vxi(stage_I_encodings_val, X_bonds_val, X_edges_val)

def sparse_reg(activ_matrix, p, beta):

        p_hat = K.mean(activ_matrix) # average over the batch samples
        #KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
        KLD = p*(K.log(p/p_hat)) + (1-p)*(K.log((1-p)/(1-p_hat)))

        return beta * K.sum(KLD) # sum over the layer units

def vni_vxi(atoms, bonds, edges):
    vni, _ = mask_atoms_by_degree(atoms,edges,bonds = bonds)
    summed_bond_features = np.sum(bonds, axis=-2)
    vxi = np.concatenate([atoms, summed_bond_features], axis=-1)
    return vni, vxi
