import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter
from model.Dipole import Dip_c, Dip_g, Dip_l, Retain
from model.PKGAT import GATModel


class DiseasePredModel(nn.Module):
    def __init__(self, dipole_type: str, input_dim, output_dim, hidden_dim, embed_dim, bi_direction=False, device=torch.device("cuda")):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.bi_direction = bi_direction
        self.device = device
        
        self.Wlstm = nn.Linear(output_dim, output_dim, bias=False)
        if dipole_type == "Dip_l":
            self.dipole = Dip_l(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif dipole_type == "Dip_c":
            self.dipole = Dip_c(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                max_timesteps=10,
                bi_direction=self.bi_direction,
                device=self.device
            )
        elif dipole_type == "Dip_g":
            self.dipole = Dip_g(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction,
                device=self.device
            )
        else: # Retain
            self.dipole = Retain(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                device=self.device
            )
        
        self.Wkg = nn.Linear(output_dim, output_dim, bias=False)
        self.pkgat = GATModel(
            nfeat=input_dim,
            nemb=self.embed_dim,
            gat_layers=1,
            gat_hid=self.output_dim,
            dropout=0.1,
            device=self.device
        )

        self.out_linear = nn.Linear(output_dim, output_dim, bias=False)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x1, visit_index, adj_index, w_index, indicator, only_dipole, p):
        """
        x1: (batch_size , 6, 2850)
        visit_index: (batch_size , 6, neighbour_size)
        adj_index: (batch_size , 6, neighbour_size)
        w_index: [batch_size * [6 * {neighbour_size}]]
        indicator: tensor(batch_size, 6 , neighbour_size)
        adj: (input_size, input_size) adjacent matrix
        """
        
        lstm_out = self.dipole(x1)
        if only_dipole == True:
            return self.out_activation(lstm_out)
        
        else:
            lstm_out = self.Wlstm(lstm_out)

            pkgat_out = self.pkgat(visit_index, adj_index, w_index, indicator)
            kg_out = self.Wkg(pkgat_out)  # (batch_size, output_dim)

            final = p*lstm_out + (1-p)*kg_out  # (batch_size, output_dim)
            final = self.out_linear(final)  # (batch_size, output_dim)

        return self.out_activation(final)