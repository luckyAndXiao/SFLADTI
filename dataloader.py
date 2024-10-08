import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein, smiles2onehot


class DTIDataset(data.Dataset):

    def __init__(self, list_IDs,  df, max_drug_nodes=289):
        self.list_IDs = list_IDs
        self.df = df
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        # todo
        v_smiles = smiles2onehot(v_d)
        # todo
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer,edge_featurizer=self.bond_featurizer)
        #Graph
        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        if num_actual_nodes < self.max_drug_nodes:
            num_virtual_nodes = self.max_drug_nodes - num_actual_nodes

            virtual_node_bit = torch.zeros([num_actual_nodes, 1])
            actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
            v_d.ndata['h'] = actual_node_feats
            virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                           torch.ones(num_virtual_nodes, 1)), 1)
            v_d.add_nodes(num_virtual_nodes, {'h': virtual_node_feat})

        v_d = v_d.add_self_loop()

        #Protein seq
        v_p = self.df.iloc[index]['Protein']
        v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]['Y']

        return v_smiles, v_d, v_p, y

#
# if __name__ == '__main__':
#     atom_featurizer = CanonicalAtomFeaturizer()
#     bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
#     fc = partial(smiles_to_bigraph, add_self_loop=True)
#     smiles = 'OC1=NN=C(CC2=CC(C(=O)N3CCN(CC3)C(=O)C3CC3)=C(F)C=C2)C2=CC=CC=C12'
#     v_d = fc(smiles=smiles, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
#     print(v_d)
#     v = v_d.ndata.pop('h')
#
#     print(v.shape[1])
#     print(v.shape)