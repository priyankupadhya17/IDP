from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.hidden_dim = 128
        
        self.gat = GAT(in_channels=1, hidden_channels=int(self.hidden_dim / 2), num_layers=2, out_channels=self.hidden_dim)
        self.g1 = GCNConv(in_channels=1, out_channels=int(self.hidden_dim / 2))
        self.g2 = GCNConv(in_channels=int(self.hidden_dim / 2), out_channels=self.hidden_dim)
        self.ReLU = ReLU(inplace=True)
        self.LeakyReLU = LeakyReLU(negative_slope=0.01, inplace=False)
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=1)
        #self.softmax = Softmax(dim=0)
        self.linear1 = Linear(in_features=1, out_features=self.hidden_dim)
        self.linear2 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
    
    
    def forward(self, x, edge_index):
        
        '''
        x = [n_nodes]
        '''
        
        #[n_nodes x hidden_dim]
        x1 = self.gat(x, edge_index)
        
        #[n_nodes x hidden_dim]
        x1 = self.linear2(x1)
        
        #[n_nodes x hidden_dim]
        x2 = self.linear1(x)
        
        #[n_nodes x hidden_dim]
        out = x2 + self.ReLU(x1)
        
        #print("encoder output")
        #print(out)
        return out
        
        

    
class Decoder(Module):
    def __init__(self, final_layer = False):
        super(Decoder, self).__init__()
        
        self.final_layer = final_layer
        
        self.const = 10
        
        self.hidden_dim = 128
        
        self.feature_dim = 1
        
        self.C = 10
        
        #dim of [x.mean(), x[starting_node], x[previous_node]]
        self.x_concat_dim = 3
        
        self.Wq = torch.FloatTensor(self.x_concat_dim * self.hidden_dim, self.hidden_dim)
        self.Wq  = nn.Parameter(self.Wq)
        self.Wq.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        self.Wk = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wk  = nn.Parameter(self.Wk)
        self.Wk.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        self.Wv = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wv  = nn.Parameter(self.Wv)
        self.Wv.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        self.Wo = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wo  = nn.Parameter(self.Wo)
        self.Wo.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        self.starting_node_W = torch.FloatTensor(self.hidden_dim)
        self.starting_node_W = nn.Parameter(self.starting_node_W)
        self.starting_node_W.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        self.previous_node_W = torch.FloatTensor(self.hidden_dim)
        self.previous_node_W = nn.Parameter(self.previous_node_W)
        self.previous_node_W.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        
        '''
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=(self.hidden_dim * 2)),
            nn.ReLU(),
            nn.Linear(in_features=(self.hidden_dim * 2), out_features=(self.hidden_dim * 4)),
            nn.ReLU(),
            nn.Linear(in_features=(self.hidden_dim * 4), out_features=self.hidden_dim)
        )
        '''
        
    
    def forward(self, X, nodes_visited, starting_node=None, previous_node=None, x=None):
        
        '''
        X = n_nodes x hidden_dim
        x = X.mean(dim=0) for 1st run, out of previous decoder for 2nd run => for both cases [hidden_dim]
        '''
        
        n_nodes = X.shape[0]
        
        if starting_node == None:
            
            #[hidden_dim]
            starting_node_features = self.starting_node_W
            previous_node_features = self.previous_node_W
        
        else:
            
            #[hidden_dim]
            starting_node_features = X[starting_node]
            previous_node_features = X[previous_node]
        
        #x = x.view(-1,)
        # [3*hidden_dim]
        features_concat = torch.cat([x, starting_node_features, previous_node_features])
        
        # [1 x 3*hidden_dim].[3*hidden_dim x hidden_dim] = [1 x hidden_dim]
        q = torch.matmul(features_concat, self.Wq)
        
        # [n_nodes x hidden_dim].[hidden_dim x hidden_dim] = [n_nodes x hidden_dim]
        k = torch.matmul(X, self.Wk)
        
        # [n_nodes x hidden_dim].[hidden_dim x hidden_dim] = [n_nodes x hidden_dim]
        v = torch.matmul(X , self.Wv)
        
        # [1 x hidden_dim].[hidden_dim x n_nodes] = [1 x n_nodes]
        u = torch.matmul(q, k.T) / math.sqrt(self.hidden_dim)
        
        
        # mask = 1 for all unvisited nodes
        mask = torch.ones(1, n_nodes)
        #mask = torch.zeros(1, n_nodes)
        for node in nodes_visited:
            mask[0][node] = 0
        
        
        if self.final_layer == True:
            
            #[1 x n_nodes]
            out = self.C * torch.tanh(u) + mask
            
            out = out.view(-1)
            
            #[n_nodes]
            out = F.softmax(out, dim=0)
            #print(mask)
            #print(out)
        
        else:
            
            #[1 x n_nodes]
            u_ = F.softmax(u + mask, dim=1)
            
            #[1 x n_nodes].[n_nodes x hidden_dim] = [1 x hidden_dim]
            h = torch.matmul(u_, v)
            
            #[1 x hidden_dim].[hidden_dim x hidden_dim] = [1 x hidden_dim]
            out = torch.matmul(h, self.Wo)
            
            #[hidden_dim]
            out = out.view(-1)
        
        return out        
        