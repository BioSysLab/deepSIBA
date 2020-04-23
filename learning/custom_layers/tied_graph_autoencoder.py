from NGF.utils import mol_shapes_to_dims
from custom_layers.tied_dense import DenseTied
import keras.backend as K
from keras.layers import Layer, add, Dense, TimeDistributed
from keras.layers import deserialize as layer_from_config
from keras.utils import deserialize_keras_object
import keras
import tensorflow as tf
from pprint import pprint
import numpy as np

def mask_atoms_by_degree(atoms, edges, bonds=None):
         
        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = K.sum(tf.keras.backend.cast(K.not_equal(edges, -1),dtype = 'float32'), axis=-1, keepdims=True)

        # For each atom, look up the features of it's neighbour
        neighbour_atom_features = neighbour_lookup(atoms, edges, include_self=False)

        # Sum along degree axis to get summed neighbour features
        summed_atom_features = K.sum(neighbour_atom_features, axis=-2)

        # Sum the edge features for each atom
        if bonds is not None:
            summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        if bonds is not None:
            summed_features = K.concatenate([summed_atom_features, summed_bond_features], axis=-1)
        else:
            summed_features = summed_atom_features

        return summed_features


def temporal_padding(x, padding=(1, 1), padvalue = 0):
  """Pads the middle dimension of a 3D tensor.
  Arguments:
      x: Tensor or variable.
      padding: Tuple of 2 integers, how many zeros to
          add at the start and end of dim 1.
  Returns:
      A padded 3D tensor.
  """
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return tf.pad(x, pattern,constant_values = padvalue)
  
  
  
  
def neighbour_lookup(atoms, edges, maskvalue=0, include_self=False):
    ''' Looks up the features of an all atoms neighbours, for a batch of molecules.

    # Arguments:
        atoms (K.tensor): of shape (batch_n, max_atoms, num_atom_features)
        edges (K.tensor): of shape (batch_n, max_atoms, max_degree) with neighbour
            indices and -1 as padding value
        maskvalue (numerical): the maskingvalue that should be used for empty atoms
            or atoms that have no neighbours (does not affect the input maskvalue
            which should always be -1!)
        include_self (bool): if True, the featurevector of each atom will be added
            to the list feature vectors of its neighbours

    # Returns:
        neigbour_features (K.tensor): of shape (batch_n, max_atoms(+1), max_degree,
            num_atom_features) depending on the value of include_self

    # Todo:
        - make this function compatible with Tensorflow, it should be quite trivial
            because there is an equivalent of `T.arange` in tensorflow.
    '''

    # The lookup masking trick: We add 1 to all indices, converting the
    #   masking value of -1 to a valid 0 index.
    masked_edges = edges + 1
    # We then add a padding vector at index 0 by padding to the left of the
    #   lookup matrix with the value that the new mask should get
    masked_atoms = temporal_padding(atoms, (1,0), padvalue=maskvalue)


    # Import dimensions
    atoms_shape = K.shape(masked_atoms)
    batch_n = atoms_shape[0]
    lookup_size = atoms_shape[1]
    num_atom_features = atoms_shape[2]

    edges_shape = K.shape(masked_edges)
    max_atoms = edges_shape[1]
    max_degree = edges_shape[2]

    # create broadcastable offset
    offset_shape = (batch_n, 1, 1)
    offset = K.reshape(tf.keras.backend.arange(stop=batch_n,start=0, dtype='int32'), offset_shape)
    offset *= lookup_size

    # apply offset to account for the fact that after reshape, all individual
    #   batch_n indices will be combined into a single big index
    flattened_atoms = K.reshape(masked_atoms, (-1, num_atom_features))
    flattened_edges = K.reshape(masked_edges + offset, (batch_n, -1))

    # Gather flattened
    flattened_result = K.gather(flattened_atoms, flattened_edges)

    # Unflatten result
    output_shape = (batch_n, max_atoms, max_degree, num_atom_features)
    output = K.reshape(flattened_result, output_shape)

    if include_self:
        return K.concatenate([tf.expand_dims(atoms, axis=2), output], axis=2)
    return output

class TiedGraphAutoencoder(Layer):
    def __init__(self, inner_layer_arg, activ, bias , init, original_atom_bond_features=None,
                 tied_to=None, encode_only=False, decode_only=False, activity_reg=None, **kwargs):
        # Initialise inner dense layers using convolution width
        # Check if inner_layer_arg is conv_width
        self.tied_to = tied_to
        self.encode_only = encode_only
        self.decode_only = decode_only
        self.bias = bias
        self.original_atom_bond_features = original_atom_bond_features
        self.activ = activ
        self.init = init
        self.reg = activity_reg

        # Case 1: check if conv_width is given
        if isinstance(inner_layer_arg, (int, np.int64)):
            self.conv_width = inner_layer_arg
            self.create_inner_layer_fn = lambda: DenseTied(self.conv_width ,
                                                            activation = self.activ ,
                                                            use_bias = bias , kernel_initializer=init,
                                                            tied_to=self.tied_to,
                                                            idx = self.idx,
                                                            activity_regularizer=self.reg,
                                                             **kwargs)
        # Case 2: Check if an initialised keras layer is given
        elif isinstance(inner_layer_arg, Layer):
            assert inner_layer_arg.built == False, 'When initialising with a keras layer, it cannot be built.'
            _, self.conv_width = inner_layer_arg.get_output_shape_for((None, None))
            # layer_from_config will mutate the config dict, therefore create a get fn
            self.create_inner_layer_fn = lambda: layer_from_config(dict(
                                                    class_name=inner_layer_arg.__class__.__name__,
                                                    config=inner_layer_arg.get_config()))
        else:
            raise ValueError('TiedAutoencoder has to be initialised with 1). int conv_width, 2). a keras layer instance, or 3). a function returning a keras layer instance.')
        
        super(TiedGraphAutoencoder, self).__init__(**kwargs)
    
    def build(self, inputs_shape):
        # Import dimensions
        (max_atoms, max_degree, num_atom_features, num_bond_features, _) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        # Add the dense layers (that contain trainable params)
        #   (for each degree we convolve with a different weight matrix)
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.inner_3D_layers = []
        self.all_layers = []
        
        self.idx = max_degree
        self_layer = self.create_inner_layer_fn()
        self_layer_type = self_layer.__class__.__name__.lower()
        self_layer.name = self.name + '_self_' + self_layer_type + '_'

        #Time Distributed layer wrapper
        self.self_3D_layer_name = self.name + '_self_timedistributed' 
        self.self_3D_layer = TimeDistributed(self_layer, name=self.self_3D_layer_name)
        if self.encode_only:
            self.self_3D_layer.build((None, max_atoms, num_atom_features+num_bond_features))
        else:
            self.self_3D_layer.build((None, max_atoms, self.conv_width))

        for degree in range(max_degree):
            self.idx = degree
            # Initialise inner layer, and rename it
            inner_layer = self.create_inner_layer_fn()
            inner_layer_type = inner_layer.__class__.__name__.lower()
            inner_layer.name = self.name + '_inner_' + inner_layer_type + '_' + str(degree)

            # Initialise TimeDistributed layer wrapper in order to parallelise
            #   dense layer across atoms (3D)
            inner_3D_layer_name = self.name + '_inner_timedistributed_' + str(degree)
            inner_3D_layer = TimeDistributed(inner_layer, name=inner_3D_layer_name)

            # Build the TimeDistributed layer (which will build the Dense layer)
            if self.encode_only:
                inner_3D_layer.build((None, max_atoms, num_bond_features+num_atom_features))
            else:
                inner_3D_layer.build((None, max_atoms, self.conv_width))

            # Store inner_3D_layer and it's weights
            self.inner_3D_layers.append(inner_3D_layer)
            self.all_layers.append(inner_3D_layer)
            if self.tied_to is not None:
                self.non_trainable_weights.append(inner_3D_layer.layer.kernel)
                if self.bias:
                    self.trainable_weights.append(inner_3D_layer.layer.bias)
            else:
                self.trainable_weights += inner_3D_layer.trainable_weights 
                    
        if self.tied_to is not None:
            self.trainable_weights.append(self.self_3D_layer.layer.bias)
            self.non_trainable_weights.append(self.self_3D_layer.layer.kernel)
        else:
            self.trainable_weights += self.self_3D_layer.trainable_weights
        
        self.all_layers.append(self_layer)
        

    def call(self, inputs, mask=None):
        atoms, bonds, edges = inputs

        if self.encode_only:
            return self.encode(inputs)
        elif self.decode_only:
            return self.decode(atoms, bonds, edges)
        else:
            return self.decode(self.encode(inputs), bonds, edges)

    def encode(self, inputs):
        atoms, bonds, edges = inputs

        # Import dimensions
        max_atoms = atoms._keras_shape[1]
        num_atom_features = atoms._keras_shape[-1]
        num_bond_features = bonds._keras_shape[-1]
        max_degree = 5
        
        # Looks up the neighbors, sums the edge features and creates vni
        summed_features, atom_degrees = self.mask_atoms_by_degree(atoms, edges, bonds)

        new_features_by_degree = self.create_layer_by_deg(max_degree, atom_degrees, 
                            (max_atoms, num_atom_features, num_bond_features), summed_features)
        zni = add(new_features_by_degree)
        summed_bonds = K.sum(bonds, axis=-2)
        vxi = K.concatenate([atoms, summed_bonds], axis=-1)
        zxi = self.self_3D_layer(vxi)

        vxi_plus_one = keras.layers.add([zni, zxi])

        return vxi_plus_one
    

    def decode(self, vxi_plus_one, bonds, edges):
        atoms = vxi_plus_one

        # Import dimensions
        max_atoms = atoms.shape[1]
        num_atom_features = atoms.shape[-1]
        num_bond_features = bonds._keras_shape[-1]
        max_degree = 5

        _, atom_degrees = self.mask_atoms_by_degree(atoms, edges, bonds=None)
        td_denses_by_degree = self.create_layer_by_deg(max_degree, atom_degrees, 
                                [max_atoms, num_atom_features, num_bond_features], vxi_plus_one)
        vni_dot = keras.layers.add(td_denses_by_degree)
        vxi_dot = self.self_3D_layer(vxi_plus_one)
        return [vni_dot, vxi_dot]


    def mask_atoms_by_degree(self, atoms, edges, bonds=None):
         
        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = K.sum(tf.keras.backend.cast(K.not_equal(edges, -1),dtype = 'float32'), axis=-1, keepdims=True)

        # For each atom, look up the features of it's neighbour
        neighbour_atom_features = neighbour_lookup(atoms, edges, include_self=False)

        # Sum along degree axis to get summed neighbour features
        summed_atom_features = K.sum(neighbour_atom_features, axis=-2)

        # Sum the edge features for each atom
        if bonds is not None:
            summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        if bonds is not None:
            summed_features = K.concatenate([summed_atom_features, summed_bond_features], axis=-1)
        else:
            summed_features = summed_atom_features

        return summed_features, atom_degrees

    def create_layer_by_deg(self, max_deg, atom_degrees, inputs, summed_features):
        # For each degree we convolve with a different weight matrix
        [max_atoms, num_atom_features, num_bond_features ]= inputs
        new_features_by_degree = []
        for degree in range(max_deg):

            # Create mask for this degree
            atom_masks_this_degree = K.cast(K.equal(atom_degrees, degree), K.floatx())

            # Multiply with hidden merge layer
            #   (use time Distributed because we are dealing with 2D input/3D for batches)
            # Add keras shape to let keras now the dimensions
            if self.encode_only:
                summed_features._keras_shape = (None, max_atoms, num_atom_features+num_bond_features)
            else:
                summed_features._keras_shape = (None, max_atoms, self.conv_width)

            new_unmasked_features = self.inner_3D_layers[degree](summed_features)
            # Do explicit masking because TimeDistributed does not support masking
            new_masked_features = new_unmasked_features * atom_masks_this_degree

            new_features_by_degree.append(new_masked_features)
        
        return new_features_by_degree

    def compute_output_shape(self, inputs_shape):

        # Import dimensions
        inputs_shape[0] = (None, int(inputs_shape[0][1]), inputs_shape[0][2])
        
        (max_atoms, _, _, _,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        if self.encode_only:
            return (num_samples, max_atoms, self.conv_width)
        else:       
            return [(num_samples, max_atoms, self.original_atom_bond_features),
                    (num_samples, max_atoms, self.original_atom_bond_features)]

class TiedGraphAutoencoderFP(Layer):
    def __init__(self, inner_layer_arg, activ, bias, init, original_atom_bond_features, tied_to=None, encode=False, decode=False,
                activity_reg=None, **kwargs):
        # Initialise 
        self.tied_to = tied_to
        self.encode = encode
        self.decode = decode
        self.original_atom_bond_features = original_atom_bond_features
        self.bias = bias
        self.reg = activity_reg

        if isinstance(inner_layer_arg, (int, np.int64)):
            self.fp_length = inner_layer_arg
            self.create_inner_layer_fn = lambda: DenseTied(self.fp_length, activation = activ ,
                                                             use_bias = bias , kernel_initializer=init, 
                                                             tied_to=self.tied_to,
                                                             idx = None,
                                                             activity_regularizer=self.reg, **kwargs) ### add inputs to dense layer
        else:
            raise ValueError('NeuralGraphHidden has to be initialised with fp_length.')

        super(TiedGraphAutoencoderFP, self).__init__(**kwargs)

    def build(self, inputs_shape):
        # Set the index for the DenseTied weight values
        # Import dimensions
        (max_atoms, _, num_atom_features, num_bond_features,
         _) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        # Add the dense layer that contains the trainable parameters
        # Initialise dense layer with specified params (kwargs) and name
        self.trainable_weights = []
        self.non_trainable_weights = []

        inner_layer = self.create_inner_layer_fn()
        inner_layer_type = inner_layer.__class__.__name__.lower()
        inner_layer.name = self.name + '_inner_'+ inner_layer_type

        # Initialise TimeDistributed layer wrapper in order to parallelise
        #   dense layer across atoms
        inner_3D_layer_name = self.name + '_inner_timedistributed'
        self.inner_3D_layer = TimeDistributed(inner_layer, name=inner_3D_layer_name)

        # Build the TimeDistributed layer (which will build the Dense layer)
        if self.encode:
            self.inner_3D_layer.build((None, max_atoms, num_atom_features+num_bond_features))
        else:
            self.inner_3D_layer.build((None, max_atoms, self.fp_length))

        # Store dense_3D_layer and it's weights

        if self.tied_to is not None:
            self.non_trainable_weights.append(self.inner_3D_layer.layer.kernel)
            if self.bias:
                self.trainable_weights.append(self.inner_3D_layer.layer.bias)
        else:
            self.trainable_weights = self.inner_3D_layer.trainable_weights


    def call(self, inputs, mask=None):
        
        if self.encode:
            return self.encoder(inputs)
        elif self.decode:
            return self.decoder(inputs)

    def encoder(self, inputs):
        atoms, bonds, edges = inputs

        final_fp_out = self.process_through_layers(atoms, bonds, edges)
        return final_fp_out

    def decoder(self, inputs):
        fp_out, _, _ = inputs

        vxi_dot = self.inner_3D_layer(fp_out)
        return vxi_dot

    def process_through_layers(self, atoms, bonds, edges):
        # Create a matrix that stores for each atom, the degree it is, use it
        #   to create a general atom mask (unused atoms are 0 padded)
        # We have to use the edge vector for this, because in theory, a convolution
        #   could lead to a zero vector for an atom that is present in the molecule
        atom_degrees = K.sum(tf.keras.backend.cast(K.not_equal(edges, -1),dtype = 'float32'), axis=-1, keepdims=True)
        general_atom_mask = K.cast(K.not_equal(atom_degrees, 0), K.floatx())

        # Sum the edge features for each atom
        summed_bond_features = K.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        atoms_bonds_features = keras.layers.Concatenate(axis=-1)([atoms, summed_bond_features])

        # Compute fingerprint
        
        fingerprint_out_unmasked = self.inner_3D_layer(atoms_bonds_features)

        # Do explicit masking because TimeDistributed does not support masking
        fingerprint_out_masked = fingerprint_out_unmasked * general_atom_mask
        final_fp_out = fingerprint_out_masked

        # Sum across all atoms
        # final_fp_out = K.sum(fingerprint_out_masked, axis=-2, keepdims = False)

        return final_fp_out

    def compute_output_shape(self, inputs_shape):

        # Import dimensions
        (max_atoms, _, _, _,
         num_samples) = mol_shapes_to_dims(mol_shapes=inputs_shape)

        if self.encode:
            return (num_samples, max_atoms, self.fp_length)
        else:
            return (num_samples, max_atoms, self.original_atom_bond_features)

    def get_config(self):
        config = super(TiedGraphAutoencoderFP, self).get_config()

        # Store config of inner layer of the 3D wrapper
        inner_layer = self.inner_3D_layer.layer
        config['inner_layer_config'] = dict(config=inner_layer.get_config(),
                                            class_name=inner_layer.__class__.__name__)
        return config

setattr(TiedGraphAutoencoder, '__deepcopy__', lambda self, _: self)
setattr(TiedGraphAutoencoderFP, '__deepcopy__', lambda self, _: self)

def create_vni_vxi(vxip1, bonds, edges):
    vxip1 = K.cast(vxip1, dtype='float32')
    vni = mask_atoms_by_degree(vxip1, edges, bonds)
    summed_bonds = K.sum(bonds, axis=-2)
    vxi = K.concatenate([vxip1, summed_bonds], axis=-1)

    return [vni, vxi]