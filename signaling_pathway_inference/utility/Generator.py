import numpy as np
import pandas as pd

def train_generator(bs,df,smiles,X_atoms, X_bonds, X_edges):
    import numpy as np
    counter=int(0)
    #Keep looping indefinetely
    while True:
        
        #Initialize batches of inputs and outputs
        ind1 = []
        ind2 = []
        
        d=[]
        
        #Keep looping until we reach batch size
        while len(ind1)<=bs: #doesn't matter if it is smi1 or smi2 since they have the same len
            
            # check to see if you reached the end of the frame
            if counter==len(df):
                counter=int(0)
                df = df.sample(frac=1).reset_index(drop=True)
            
            smi1=df['rdkit.x'][counter]
            smi2=df['rdkit.y'][counter]
            ind1.append(smiles.index(smi1))
            ind2.append(smiles.index(smi2))
            d.append(df.value[counter]/2)
            counter+=1
            
        atom_1=np.array(X_atoms[ind1],dtype = 'float32')
        bond_1=np.array(X_bonds[ind1],dtype = 'float32')
        edge_1=np.array(X_edges[ind1],dtype = 'int32')
        atom_2=np.array(X_atoms[ind2],dtype = 'float32')
        bond_2=np.array(X_bonds[ind2],dtype = 'float32')
        edge_2=np.array(X_edges[ind2],dtype = 'int32')
        
        # yield the batch to the calling function
        yield ({'atom_inputs_1':atom_1,'bond_inputs_1':bond_1,'edge_inputs_1':edge_1,'atom_inputs_2':atom_2,
                'bond_inputs_2':bond_2,'edge_inputs_2':edge_2},np.array(d,dtype = 'float32'))

def preds_generator(bs,df_cold,smiles_cold,X_atoms_cold, X_bonds_cold, X_edges_cold,siamese_net):
    
    import numpy as np
    counter=int(0)
    #Keep looping indefinetely
    while counter<len(df_cold):
        
        #Initialize batches of inputs and outputs
        ind1 = []
        ind2 = []
        
        
        #Keep looping until we reach batch size
        while len(ind1)<=bs: #doesn't matter if it is smi1 or smi2 since they have the same len
            
            # check to see if you reached the end of the frame
            if counter==len(df_cold):
                break
                
            smi1=df_cold['rdkit.x'][counter]
            smi2=df_cold['rdkit.y'][counter]
            ind1.append(smiles_cold.index(smi1))
            ind2.append(smiles_cold.index(smi2))
            counter+=1
    
            
        atom_1=np.array(X_atoms_cold[ind1],dtype = 'float32')
        bond_1=np.array(X_bonds_cold[ind1],dtype = 'float32')
        edge_1=np.array(X_edges_cold[ind1],dtype = 'int32')
        atom_2=np.array(X_atoms_cold[ind2],dtype = 'float32')
        bond_2=np.array(X_bonds_cold[ind2],dtype = 'float32')
        edge_2=np.array(X_edges_cold[ind2],dtype = 'int32')
        
        y_pred=siamese_net.predict([atom_1,bond_1,edge_1,atom_2,bond_2,edge_2],batch_size=bs)
        
        yield(y_pred)
