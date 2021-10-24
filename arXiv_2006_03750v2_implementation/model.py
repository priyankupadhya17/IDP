from torch.nn import Linear, ReLU, Module, Sequential, Softmax
from torch_geometric.nn import Sequential, GAT, GATConv
import torch
from torch.autograd import Variable


class Model(Module):
    def __init__(self, hidden_dim, feature_dim):
        super(Model, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.const = 1
        
        
        #ENCODER
        #self.gat_conv1 = GATConv(in_channels=1, out_channels=3, add_self_loops=True)
        #self.gat_conv2 = GATConv(in_channels=3, out_channels=3, add_self_loops=True)
        #self.gat_conv3 = GATConv(in_channels=3, out_channels=1, add_self_loops=True)
        self.gat = GAT(in_channels=1, hidden_channels=3, num_layers=3, out_channels=1)
        
        self.phi1 = Variable(torch.randn(hidden_dim, feature_dim), requires_grad=True)
        self.phi2 = Variable(torch.randn(hidden_dim, feature_dim), requires_grad=True)
        
    
    def decoderFunc(self, starting_node, x, edge_index):
        
        edge_index_sources = torch.where(edge_index[0] == starting_node)
        edge_index_dest = edge_index[1][edge_index_sources]
        
        alphas = (self.phi1 @ x[starting_node]).T @ (self.phi2 @ x[edge_index_dest].T)
        alphas = alphas / torch.sqrt(torch.tensor(self.hidden_dim))
        alphas = self.const * torch.tanh(alphas)
        
        alpha_probs = Softmax(dim=1)(alphas)
        
        next_starting_node = edge_index_dest[alpha_probs.argmax(dim=1)]
        
        return next_starting_node

    
    def forward(self, x, edge_index):
        
        #x = self.gat_conv1(x, edge_index)
        #x = self.gat_conv2(x, edge_index)
        #x = self.gat_conv3(x, edge_index)
        x = self.gat(x, edge_index)
        x = Softmax(dim=0)(x)
        print(x)
        
        n_nodes = x.shape[0]
        starting_node = torch.argmax(x, dim=0)
        
        
        for i in range(n_nodes):
            print(starting_node)
            starting_node = self.decoderFunc(starting_node, x, edge_index)
        
        
        return x

