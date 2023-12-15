import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parameter import Parameter
from model.Dipole import Dip_c, Dip_g, Dip_l, Retain
from model.PKGAT import GATModel


class DiseasePredModel(nn.Module):
    def __init__(self, dipole_type: str, input_dim, output_dim, hidden_dim, bi_direction=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        
        self.Wlstm = nn.Linear(output_dim, output_dim, bias=False)
        if dipole_type == "Dip_l":
            self.dipole = Dip_l(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction
            )
        elif dipole_type == "Dip_c":
            self.dipole = Dip_c(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                max_timesteps=10,
                bi_direction=self.bi_direction
            )
        elif dipole_type == "Dip_g":
            self.dipole = Dip_g(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                bi_direction=self.bi_direction
            )
        else: # Retain
            self.dipole = Retain(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim
            )
        
        self.Wkg = nn.Linear(output_dim, output_dim, bias=False)
        self.pkgat = GATModel(
            nfeat=input_dim,
            nemb=self.hidden_dim,
            gat_layers=1,
            gat_hid=self.output_dim,
            dropout=0.1 
        )

        self.out_linear = nn.Linear(output_dim, output_dim, bias=False)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x, adj):
        """
        x: (batch_size , seq_length(visit_num), input_size)
        adj: (input_size, input_size) adjacent matrix
        """
        lstm_out = self.dipole(x)  # (batch_size, output_dim)
        lstm_out = self.Wlstm(lstm_out)  # (batch_size, output_dim)

        # 由于这里不能使用批处理，所以需要拆开处理batch
        pkgat_out = []
        for batch_i in x:
            # batch_i: (seq_length(visit_num), input_size)
            pkgat_out_i = self.pkgat(
                batch_i, adj
            )  # (output_dim)
            pkgat_out.append(torch.unsqueeze(pkgat_out_i, dim=0)) # (1, output_dim)
        pkgat_out = torch.cat(pkgat_out, dim=0)  # (batch_size, output_dim)

        kg_out = self.Wkg(pkgat_out)  # (batch_size, output_dim)

        final = lstm_out + kg_out  # (batch_size, output_dim)
        final = self.out_linear(final)  # (batch_size, output_dim)

        return self.out_activation(final)


        
        
        
