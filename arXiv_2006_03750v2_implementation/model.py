from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv, SAGEConv
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
        
        self.gat1 = GAT(in_channels=1, hidden_channels=int(self.hidden_dim / 2), num_layers=3, out_channels=self.hidden_dim)
        self.gat2 = GAT(in_channels=self.hidden_dim, hidden_channels=self.hidden_dim, num_layers=3, out_channels=self.hidden_dim)
        
        self.ReLU = ReLU(inplace=True)
        
        self.linear1 = Linear(in_features=1, out_features=self.hidden_dim)
        self.linear2 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.linear3 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.linear4 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
    
    
    def forward(self, x, edge_index):
        
        '''
        x = [n_nodes]
        '''
        
        #[n_nodes x hidden_dim]
        x1 = self.linear1(x)
        
        #[n_nodes x hidden_dim]
        x2 = self.gat1(x, edge_index)
        
        #[n_nodes x hidden_dim]
        x2 = self.linear2(x2)
        
        #[n_nodes x hidden_dim]
        x2 = x1 + self.ReLU(x2)
        
        #[n_nodes x hidden_dim]
        x3 = self.gat2(x2, edge_index)
        
        #[n_nodes x hidden_dim]
        x3 = self.linear3(x3)
        
        #[n_nodes x hidden_dim]
        out = x2 + self.ReLU(x3)
        
        out = self.linear4(out)
        
        
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

    
class Decoder_Entropy(Module):
    def __init__(self):
        super(Decoder_Entropy, self).__init__()
        
        self.hidden_dim = 128
        
        self.ReLU = ReLU(inplace=True)
        
        self.sigmoid = Sigmoid()
        
        self.linear1 = Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2))
        self.linear2 = Linear(in_features=int(self.hidden_dim / 2), out_features=int(self.hidden_dim / 4))
        self.linear3 = Linear(in_features=int(self.hidden_dim / 4), out_features=1)
    
    
    def forward(self, x):
        
        '''
        x = [n_nodes, hidden_dim]
        '''
        
        x = self.linear1(x)
        x = self.ReLU(x)
        
        x = self.linear2(x)
        x = self.ReLU(x)
        
        x = self.linear3(x)
        x = self.sigmoid(x)
        
        
        #print(out)
        return x
