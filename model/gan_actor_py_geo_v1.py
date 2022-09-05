import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GATConv
from model.gan_layer_py_geo import GATLayer
### https://github.com/marblet/GNN_models_pytorch_geometric


class GAT_actor(nn.Module):
    #c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2
    def __init__(self, nfeat, nhid, num_heads, nclass, dropout):
        super(GAT_actor, self).__init__()
        self.gc1 = GATLayer(nfeat, nhid, num_heads=num_heads, concat_heads=True, alpha=0.2)
        self.gc2 = GATLayer(nhid, nclass, num_heads=num_heads, concat_heads=True, alpha=0.2)
        self.dropout = dropout

    # def reset_parameters(self):
    #     self.gc1.reset_parameters()
    #     self.gc2.reset_parameters()

    def forward(self, obs, adj_matrix):
        #node_feats, adj_matrix, print_attn_probs = False
        #x = obs
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.dropout(obs, self.dropout, training=self.training)
        activation1 = self.gc1(activation1, adj_matrix, print_attn_probs=False)

        activation1 = F.elu(activation1)
        activation2 = self.gc2(activation1, adj_matrix, print_attn_probs=False)

        shape = activation2.shape[0]

        output = F.softmax(activation2, dim=2)
        if shape == 1:
            output = torch.squeeze(output)
            return output
        else:
            return output




"""
class GCN_actor(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):

        super(GCN_actor, self).__init__()
        self.layer1 = GCNConv(nfeat, nhid)
        self.layer2 = GCNConv(nhid, nhid)
        # self.layer3 = GCNConv(nhid, nhid)
        # self.layer4 = GCNConv(nhid, nhid)
        # self.layer5 = GCNConv(nhid, nhid)
        # self.layer6 = GCNConv(nhid, nhid)
        # self.layer7 = GCNConv(nhid, nclass)
        self.layer3 = GCNConv(nhid, nclass)
        self.dropout = dropout


    def forward(self, obs, adj):

        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.tanh(self.layer1(obs, adj))
        activation1 = F.dropout(activation1, self.dropout, training=self.training)

        activation2 = F.tanh(self.layer2(activation1, adj))
        activation2 = F.dropout(activation2, self.dropout, training=self.training)

        # activation3 = F.tanh(self.layer3(activation2, adj))
        # activation3 = F.dropout(activation3, self.dropout, training=self.training)
        #
        # activation4 = F.tanh(self.layer4(activation3, adj))
        # activation4 = F.dropout(activation4, self.dropout, training=self.training)
        #
        # activation5 = F.tanh(self.layer5(activation4, adj))
        # activation5 = F.dropout(activation5, self.dropout, training=self.training)
        #
        # activation6 = F.tanh(self.layer6(activation5, adj))
        # activation6 = F.dropout(activation6, self.dropout, training=self.training)
        #
        # activation7 = self.layer7(activation6, adj)
        # shape_dim = activation7.dim()

        activation3 = self.layer3(activation2, adj)
        shape_dim = activation3.dim()



        if shape_dim == 2:
            #output = F.softmax(activation7, dim=1)
            output = F.softmax(activation3, dim=1)
        else:
            #output = F.softmax(activation7, dim=2)
            output = F.softmax(activation3, dim=2)
        return output 
"""