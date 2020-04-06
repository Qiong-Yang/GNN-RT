# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 09:24:09 2019

@author: qy
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from tqdm import tqdm

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    """Get the adjacency from mol"""
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)
    

if __name__ == "__main__":
    import pandas as pd
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.Crippen import MolLogP
    from rdkit.Chem.rdmolops import GetFormalCharge
    from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds
    from mordred.LogS import LogS
    from getf import getf
    CalcLogS = LogS()
    
    def preprocess(data, dir_input):
        """Get molecular fingerprints,adjacencies and descriptors from it's inchi 
        and Store these values on the computer"""
        
        smiles = list(data['inchi'])
        rts = list(data['rt'])
        
        
        Smiles, molecules, adjacencies, properties,descriptors = '', [], [], [],[]
        for i, inc in enumerate(smiles):
            mol = Chem.MolFromInchi(inc)
            
            if mol is None:
                continue
            else:
                smi = Chem.MolToSmiles(mol)
            mol = Chem.AddHs(mol)
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
            
            adjacency = create_adjacency(mol)
        
            Smiles += smi + '\n'
            molecules.append(fingerprints)
            adjacencies.append(adjacency)
            properties.append([[rts[i]]])
            a = getf(mol)
            """Select 65 molecular descriptors from the 200 molecular descriptors getting from getf.py"""
            descriptors.append([a['rdkit']['MolWt'],a['rdkit']['MaxAbsPartialCharge'] ,a['rdkit']['MaxEStateIndex'] , a['rdkit']['MinAbsEStateIndex'] ,
                                a['rdkit']['MaxPartialCharge'],a['rdkit']['MinAbsPartialCharge'] ,a['rdkit']['MolMR'] ,a['rdkit']['PEOE_VSA1'] ,
                                a['rdkit']['PEOE_VSA2'] ,a['rdkit']['PEOE_VSA3'] ,a['rdkit']['PEOE_VSA4'] ,a['rdkit']['PEOE_VSA5'] ,a['rdkit']['PEOE_VSA6'] ,
                                a['rdkit']['PEOE_VSA7'],a['rdkit']['PEOE_VSA8'],a['rdkit']['PEOE_VSA9'] , a['rdkit']['PEOE_VSA10'],   a['rdkit']['PEOE_VSA11'] ,
                                a['rdkit']['PEOE_VSA12'] ,a['rdkit']['PEOE_VSA13'],a['rdkit']['SlogP_VSA1'] ,a['rdkit']['SlogP_VSA2'] ,
                                a['rdkit']['SlogP_VSA3'] ,a['rdkit']['SlogP_VSA4'], a['rdkit']['SlogP_VSA5'] ,a['rdkit']['SlogP_VSA6'] ,
                                a['rdkit']['SlogP_VSA7'] ,a['rdkit']['SlogP_VSA8'] ,a['rdkit']['SlogP_VSA9'] ,a['rdkit']['SlogP_VSA10'], 
                                a['rdkit']['SlogP_VSA11'] ,a['rdkit']['SlogP_VSA12'] ,a['rdkit']['SMR_VSA1'] ,a['rdkit']['SMR_VSA2'] ,
                                a['rdkit']['SMR_VSA3'] ,a['rdkit']['SMR_VSA4'] ,a['rdkit']['SMR_VSA5'] ,a['rdkit']['SMR_VSA6'] ,
                                a['rdkit']['SMR_VSA7'] ,a['rdkit']['SMR_VSA8'] ,a['rdkit']['SMR_VSA9'] ,a['rdkit']['SMR_VSA10'],
                                a['rdkit']['TPSA'],a['rdkit']['LabuteASA'],a['rdkit']['MolLogP'],a['rdkit']['VSA_EState1'], a['rdkit']['VSA_EState2'], 
                                a['rdkit']['VSA_EState3'] ,a['rdkit']['VSA_EState4']  ,a['rdkit']['VSA_EState5']  ,a['rdkit']['VSA_EState6']  ,
                                a['rdkit']['VSA_EState7']  ,a['rdkit']['VSA_EState8'],  a['rdkit']['VSA_EState9'] , a['rdkit']['VSA_EState10'],
                                a['rdkit']['qed'],a['rdkit']['MinEStateIndex'],   a['rdkit']['Kappa1'],a['rdkit']['Kappa2'],  a['rdkit']['Kappa3'] ,  
                                a['rdkit']['FractionCSP3'],   a['rdkit']['HallKierAlpha'],  a['rdkit']['BalabanJ'],a['rdkit']['BertzCT'],   a['rdkit']['Chi1']])
            
           """Regularization of descriptors and properties"""
        from sklearn.preprocessing import normalize
        descriptors = normalize(descriptors, axis=1, norm='max')                                        
        properties = np.array(properties)
        mean, std = np.mean(properties), np.std(properties)
        properties = np.array((properties - mean) / std)

    
        os.makedirs(dir_input, exist_ok=True)

        with open(dir_input + 'Smiles.txt', 'w') as f:
            f.write(Smiles)
        np.save(dir_input + 'molecules', molecules)
        np.save(dir_input + 'adjacencies', adjacencies)
        np.save(dir_input + 'properties', properties)
        np.save(dir_input + 'descriptors', descriptors)
        np.save(dir_input + 'mean', mean)
        np.save(dir_input + 'std', std)
        dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
      
    radius =1
      
    """load data"""
    data = pd.read_table('data/SMRT_dataset.csv',sep=';')
    """drop the molecule with retention time < 300 seconds"""
    for i in range(len(data['rt'])):
        if data['rt'][i]<=300:
            data.drop(index=i,inplace=True)     
    dir_input = ('data/Input/radius' + str(radius)+'/')
    
    preprocess(data, dir_input)
    print('The preprocess of the dataset has finished!') 
    
