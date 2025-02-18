from __future__ import print_function, division
import torch
import torch.nn as nn

class CONVOLUTE_LAYER(nn.Module):
    def __init__(self, atom_fea_len, nbr_fea_len):
        super(CONVOLUTE_LAYER, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len, 2*self.atom_fea_len) 
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        N, M = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat([atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len), atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter) 
        nbr_core = self.softplus1(nbr_core)
        nbr_summed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_summed = self.bn2(nbr_summed)
        out = self.softplus2(atom_in_fea + nbr_summed)
        return out

class GraphNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len, sputter_fea_len, atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1):
        super(GraphNet, self).__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CONVOLUTE_LAYER(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)]) 
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)]) 

        self.addlayer_1 = nn.Linear(145, 200)
        self.addnl_1 = nn.ReLU()
        self.addlayer_2 = nn.Linear(200, 400)
        self.addnl_2 = nn.ReLU()
        self.addlayer_3 = nn.Linear(400, 800)
        self.addnl_3 = nn.ReLU()

        self.fc_out = nn.Linear(800, 1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, sputter_fea, crystal_atom_idx): 
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses): # h_fea_len
                crys_fea = softplus(fc(crys_fea)) # 20 x 128 
                print(crys_fea.shape)

        crys_fea_cat = torch.cat((crys_fea, sputter_fea), dim=1)
        print(crys_fea_cat.shape)

        crys_fea_cat = self.addlayer_1(crys_fea_cat)
        crys_fea_cat = self.addnl_1(crys_fea_cat)
        crys_fea_cat = self.addlayer_2(crys_fea_cat)
        crys_fea_cat = self.addnl_2(crys_fea_cat)
        crys_fea_cat = self.addlayer_3(crys_fea_cat)
        crys_fea_cat = self.addnl_3(crys_fea_cat)

        out = self.fc_out(crys_fea_cat)
        return out

    def pooling(self, atom_fea, crystal_atom_idx):
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True) for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)