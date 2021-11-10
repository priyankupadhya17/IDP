from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch import nn


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.gat = GAT(in_channels=1, hidden_channels=3, num_layers=3, out_channels=1)
        self.g1 = GCNConv(in_channels=1, out_channels=3)
        self.g2 = GCNConv(in_channels=3, out_channels=1)
        self.ReLU = ReLU(inplace=True)
        self.LeakyReLU = LeakyReLU(negative_slope=0.01, inplace=False)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=0)
    
    
    def forward(self, x, edge_index):
        #print("features to encode")
        #print(x)
        
        #x = self.gat(x, edge_index)
        x = self.g1(x, edge_index)
        x = self.ReLU(x)
        x = self.g2(x, edge_index)
        x = self.softmax(x)
        
        #print("gat output")
        #print(x)
        return x

    
class Decoder(Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.const = 10
        
        self.hidden_dim = 64
        self.feature_dim = 1
        
        self.C = 10
        
        #dim of [x.mean(), x[starting_node], x[previous_node]]
        self.x_concat_dim = 3
        
        self.Wq = Parameter(torch.randn(self.x_concat_dim, self.hidden_dim))
        self.Wk = Parameter(torch.randn(self.feature_dim, self.hidden_dim))
        self.Wv = Parameter(torch.randn(self.feature_dim, self.hidden_dim))
        
        self.Wo = Parameter(torch.randn(self.feature_dim, self.hidden_dim))
        
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=(self.hidden_dim * 2)),
            nn.ReLU(),
            nn.Linear(in_features=(self.hidden_dim * 2), out_features=(self.hidden_dim * 4)),
            nn.ReLU(),
            nn.Linear(in_features=(self.hidden_dim * 4), out_features=self.hidden_dim)
        )
        
    
    def forward(self, x, nodes_visited, starting_node=None, previous_node=None):
        
        n_nodes = x.shape[0]
        
        #print(f"nodes_visited = {nodes_visited}")
        
        if starting_node == None:
            starting_node = x.argmax(dim=0)
        
        if previous_node == None:
            previous_node = starting_node
        
        x_features_mean = torch.mean(x).unsqueeze(dim=0)
        starting_node_features = x[starting_node]
        previous_node_features = x[previous_node]
        
        # [1x3]
        features_concat = torch.cat([x_features_mean, starting_node_features, previous_node_features])
        
        # [1x3].[3x16] = [1x16]
        q = torch.matmul(features_concat, self.Wq)
        
        # [n_nodes x 1].[1 x 16] = [n_nodes x 16]
        k = torch.matmul(x, self.Wk)
        
        # [n_nodes x 1].[1 x 16] = [n_nodes x 16]
        v = torch.matmul(x , self.Wv)
        
        # [1 x 16].[16 x n_nodes] = [1 x n_nodes]
        u = torch.matmul(q, k.T) / math.sqrt(self.hidden_dim)
        
        mask = torch.ones(1, n_nodes)
        for node in nodes_visited:
            mask[0][node] = 0
        
        
        #"""
        # a = [1 x n_nodes]
        #a = F.softmax(u + mask, dim=1)
        #a = F.softmax(u, dim=0)
        a = torch.tanh(u + mask)
        
        # [1x n_nodes].[n_nodes x 16] = [1 x 16]
        h = torch.matmul(a, v)
        
        #[16]
        h = h.squeeze()
        
        #[16]
        out = self.linear(h)
        
        #[1 x 16]
        out = torch.unsqueeze(out, dim=0)
        
        # [n_nodes x 1].[1 x 16] = [n_nodes x 16]
        wts = torch.matmul(x, self.Wo)
        
        # [n_nodes x 16].[16 x 1] = [n_nodes x 1]
        out = torch.matmul(wts, out.T)
        
        #[n_nodes]
        out = out.squeeze()
        
        probs = F.softmax(out + mask.squeeze(), dim=0)
        #"""
        
        """
        
        #[1 x n_nodes]
        probs = self.C * torch.tanh(u) + mask
        
        #[n_nodes]
        probs = probs.squeeze()
        
        probs = F.softmax(probs, dim=0)
        
        """
        
        #print(probs)
        
        next_node = probs.argmax(dim=0)
        return probs, next_node
        
        
        