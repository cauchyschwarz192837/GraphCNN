from __future__ import print_function, division
import csv
import json
import os
import random
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_val_test_loader(dataset, collate_fn=default_collate, batch_size=20, train_ratio=None, val_ratio=0.2, test_ratio=0.2, return_test=False):

    # CHECK!!! 

    total_size = len(dataset)
    if train_ratio == None:
        train_ratio = 1 - val_ratio - test_ratio
    indices = list(range(total_size))
    train_size = int(train_ratio * total_size) 
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, collate_fn=collate_fn)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=1, collate_fn=collate_fn)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=1, collate_fn=collate_fn)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

def mynewCOLLATE(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_sputter_fea, crystal_atom_idx, batch_target, batch_cif_ids = [], [], [], [], [], [], []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx, sputter_fea), target, cif_id) in enumerate(dataset_list): # ref last... line
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx+base_idx) 
        batch_sputter_fea.append(sputter_fea)
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0), torch.cat(batch_nbr_fea, dim=0), torch.cat(batch_nbr_fea_idx, dim=0), torch.cat(batch_sputter_fea, dim=0), crystal_atom_idx), torch.stack(batch_target, dim=0), batch_cif_ids

class GAUSSIANKERNELTHING(object):
    def __init__(self, dmin, dmax, step, vari=None):
        self.filter = np.arange(dmin, dmax+step, step)
        if vari is None:
            vari = step
        self.vari = vari
    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.vari**2) # need check function!!!


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}
    def get_atom_fea(self, atom_type):
        return self._embedding[atom_type]
    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
    def state_dict(self):
        return self._embedding
    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
        return self._decodedict[idx]

class INITIALISE_FROM_JSON(AtomInitializer):
    def __init__(self, embedding_file):
        with open(embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(INITIALISE_FROM_JSON, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CIFData(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.sputter_params = np.loadtxt(r'/Users/admin/Downloads/GNN/gnn/gnn/sputter_data.csv', delimiter=',')
        
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        self.ari = INITIALISE_FROM_JSON(atom_init_file)
        self.gdf = GAUSSIANKERNELTHING(dmin=dmin, dmax=self.radius, step=step)
    def __len__(self):
        return len(self.id_prop_data)
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id+'.cif'))
        sputter_fea = self.sputter_params[int(cif_id), :]
        sputter_fea = np.expand_dims(sputter_fea, 0) # dimensions not Matching !!!

        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        sputter_fea = torch.Tensor(sputter_fea)

        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs] # sort to get increasing order of distance

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
 
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        target = torch.Tensor([float(target)])
        return (atom_fea, nbr_fea, nbr_fea_idx, sputter_fea), target, cif_id