import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GATConv
from model.gan_layer_py_geo import GATLayer
### https://github.com/marblet/GNN_models_pytorch_geometric


class GAT_actor(nn.Module):
    #c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2
    def __init__(self, nfeat, nhid_gan, num_heads, nclass_gan, in_dim_spec, nhid_fc, nclass_f, nhid_f, dropout):
        super(GAT_actor, self).__init__()
        self.layer_gan_1 = GATLayer(nfeat, nhid_gan, num_heads=num_heads, concat_heads=True, alpha=0.2)
        self.layer_gan_2 = GATLayer(nhid_gan, nhid_gan, num_heads=num_heads, concat_heads=True, alpha=0.2)
        self.layer_gan_3 = GATLayer(nhid_gan, nclass_gan * num_heads, num_heads=num_heads, concat_heads=True, alpha=0.2)

        self.layer_fc_1 = nn.Linear(in_dim_spec, nhid_fc)
        self.layer_fc_2 = nn.Linear(nhid_fc + nclass_gan * num_heads * 9, nhid_f)
        self.layer_fc_3 = nn.Linear(nhid_f, nclass_f * 7)

        self.dropout = dropout

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     self.gc2.reset_parameters()

    def forward(self, obs_gan, obs_fc, adj_matrix):
        #node_feats, adj_matrix, print_attn_probs = False
        #x = obs
        if isinstance(obs_gan, np.ndarray):
            obs_gan = torch.tensor(obs_gan, dtype=torch.float)

        if isinstance(obs_fc, np.ndarray):
            obs_fc = torch.tensor(obs_fc, dtype=torch.float)

        activation_gan_1 = F.dropout(obs_gan, self.dropout, training=self.training)
        activation_gan_1 = self.layer_gan_1(activation_gan_1, adj_matrix, print_attn_probs=False)
        activation_gan_1 = F.elu(activation_gan_1)

        activation_gan_2 = F.dropout(activation_gan_1, self.dropout, training=self.training)
        activation_gan_2 = self.layer_gan_2(activation_gan_2, adj_matrix, print_attn_probs=False)
        activation_gan_2 = F.elu(activation_gan_2)

        activation_gan_3 = F.dropout(activation_gan_2, self.dropout, training=self.training)
        activation_gan_3 = self.layer_gan_3(activation_gan_3, adj_matrix, print_attn_probs=False)
        activation_gan_3 = F.elu(activation_gan_3)

        activation_fc_1 = torch.tanh(self.layer_fc_1(obs_fc))

        shape = activation_gan_3.shape[0]
        #output = F.softmax(activation2, dim=2)
        if shape == 1:
            activation_gan_3 = torch.squeeze(activation_gan_3)
            activation_gan_3 = torch.flatten(activation_gan_3)
            activation_fc_2 = torch.cat((activation_gan_3, activation_fc_1), dim=-1)
            activation_fc_2 = F.dropout(activation_fc_2, self.dropout, training=self.training)
            activation_fc_2 = torch.tanh(self.layer_fc_2(activation_fc_2))
            output = F.softmax(torch.reshape(self.layer_fc_3(activation_fc_2), (7, 3)), dim=1)
        else:
            activation_gan_3 = torch.flatten(activation_gan_3, start_dim=1)
            activation_fc_2 = torch.cat((activation_gan_3, activation_fc_1), dim=1)
            activation_fc_2 = F.dropout(activation_fc_2, self.dropout, training=self.training)
            activation_fc_2 = torch.tanh(self.layer_fc_2(activation_fc_2))
            output = F.softmax(torch.reshape(self.layer_fc_3(activation_fc_2), (shape, 7, 3)), dim=2)
        return output


