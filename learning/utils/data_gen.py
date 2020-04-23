from __future__ import absolute_import, division
import numpy as np
import pandas as pd
import pickle
import tables
import csv
import os
import re
import csv

def total_exp(cell_lines, path_to_networks):
    for cl in cell_lines:
        nets = 0
        for exp in os.listdir(path_to_networks + cl):
            path_exp = path_to_networks + cl + '/' + exp
            if os.path.isdir(path_exp + '/Results_CARNIVAL'):
                nets += 1
        print(f'Cell line {cl} has a total of {len(os.listdir(path_to_networks + cl))} experiments and {nets} CARNIVAL processed experiments.')

def count_models(files):
    count = 0
    for f in files:
        inter = re.search('\A[i].*', f)
        if inter is not None:
            count += 1
    return count

def node_bond_mat(reader):
    """
    Takes the tsv file and outputs the connectivity matrices.
    """

    node1 = []
    node2 = []
    bonds = []
    node_dict = {}
    
    for row in reader:
        if (row['Node1'] != 'Perturbation') and (row['Node2'] != 'Perturbation'):
            node1.append(row['Node1'])
            node2.append(row['Node2'])
            bonds.append(int(row['Sign']))
       
    for idx, node in enumerate(np.unique(node1 + node2)):
        node_dict[node] = idx
        max_atoms = idx
        
    node1 = [node_dict[node] for node in node1]
    node2 = [node_dict[node] for node in node2]
    
    return node1, node2, bonds, max_atoms

def connectivity_mat(reader, max_atoms, max_degree, num_bond_features):
    """
    Returns the edges and bonds matrices of the input signaling network.
    """

    edge_mat = np.full((max_atoms, max_degree), fill_value=-1)
    bonds = np.full((max_atoms, max_degree), fill_value=0)
    
    node1, node2, bonds_mat, _ = node_bond_mat(reader)
 
    for n1, b, n2 in zip(node1, bonds_mat, node2):
        for i in range(max_degree):
            if edge_mat[n1][i]==-1:
                edge_mat[n1][i]=n2
                bonds[n1][i] = b
                break
                
        for i in range(max_degree):       
            if edge_mat[n2][i] == -1:
                edge_mat[n2][i] = n1
                bonds[n2][i] = b
                break
    
    return edge_mat, bonds

def node_attributes(prot_dict):
    pass

def gather(self, dim, index):
    """
    Gathers values along an axis specified by ``dim``.

    For a 3-D tensor the output is specified by:
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

    Parameters
    ----------
    dim:
        The axis along which to index
    index:
        A tensor of indices of elements to gather

    Returns
    -------
    Output Tensor
    """
    idx_xsection_shape = index.shape[:dim] + \
        index.shape[dim + 1:]
    self_xsection_shape = self.shape[:dim] + self.shape[dim + 1:]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError("Except for dimension " + str(dim) +
                         ", all dimensions of index and self should be the same size")
    if index.dtype != np.dtype('int_'):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(self, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)

def temporal_padding(x, padding=(1, 1), padvalue = 0):
  """Pads the middle dimension of a 3D array.
  Arguments:
      x: array or variable.
      padding: Tuple of 2 integers, how many zeros to
          add at the start and end of dim 1.
  Returns:
      A padded 3D array.
  """
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return np.pad(x, pattern, mode='constant', constant_values = padvalue)

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
    atoms_shape = masked_atoms.shape
    batch_n = atoms_shape[0]
    lookup_size = atoms_shape[1]
    num_atom_features = atoms_shape[2]

    edges_shape = masked_edges.shape
    max_atoms = edges_shape[1]
    max_degree = edges_shape[2]

    # create broadcastable offset
    offset_shape = (batch_n, 1, 1)
    offset = np.reshape(np.arange(start=0, stop=batch_n, dtype='int32'), offset_shape)
    offset *= lookup_size

    # apply offset to account for the fact that after reshape, all individual
    #   batch_n indices will be combined into a single big index
    flattened_atoms = np.reshape(masked_atoms, (-1, num_atom_features))
    flattened_edges = np.reshape(masked_edges + offset, (batch_n, -1))

    # Gather flattened
    flattened_result = np.take(flattened_atoms, flattened_edges, axis=0)

    # Unflatten result
    output_shape = (batch_n, max_atoms, max_degree, num_atom_features)
    output = np.reshape(flattened_result, output_shape)

    if include_self:
        return np.concatenate([np.expand_dims(atoms, axis=2), output], axis=2)
    return output


def mask_atoms_by_degree(atoms, edges, bonds=None):
         
        # Create a matrix that stores for each atom, the degree it is
        atom_degrees = np.sum(np.ndarray.astype(np.not_equal(edges, -1),dtype = 'float32'), axis=-1, keepdims=True)

        # For each atom, look up the features of it's neighbour
        neighbour_atom_features = neighbour_lookup(atoms, edges, include_self=False)

        # Sum along degree axis to get summed neighbour features
        summed_atom_features = np.sum(neighbour_atom_features, axis=-2)

        # Sum the edge features for each atom
        if bonds is not None:
            summed_bond_features = np.sum(bonds, axis=-2)

        # Concatenate the summed atom and bond features
        if bonds is not None:
            summed_features = np.concatenate([summed_atom_features, summed_bond_features], axis=-1)
        else:
            summed_features = summed_atom_features

        return summed_features, atom_degrees