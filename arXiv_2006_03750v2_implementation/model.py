from torch.nn import Linear, ReLU, Module, Sequential, Softmax
from torch_geometric.nn import Sequential, GAT, GATConv
import torch
from torch.autograd import Variable


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.gat = GAT(in_channels=1, hidden_channels=6, num_layers=3, out_channels=1)
    
    
    def forward(self, x, edge_index):
        
        x = self.gat(x, edge_index)
        x = Softmax(dim=0)(x)
        
        return x

    
class Decoder():
    def __init__(self, hidden_dim, feature_dim):
        super(Decoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.const = Variable(torch.randint(low=1,high=5,size=(1,1)))
        
        
        self.phi1 = Variable(torch.randn(hidden_dim, feature_dim), requires_grad=True)
        self.phi2 = Variable(torch.randn(hidden_dim, feature_dim), requires_grad=True)
        
    
    def decode(self, starting_node, x, edge_index, node_set, line_graph_nodes):
        
        edge_index_sources = torch.where(edge_index[0] == starting_node)
        edge_index_dest = edge_index[1][edge_index_sources]
        
        alphas = (self.phi1 @ x[starting_node]).T @ (self.phi2 @ x[edge_index_dest].T)
        alphas = alphas / torch.sqrt(torch.tensor(self.hidden_dim))
        alphas = self.const * torch.tanh(alphas)
        
        alpha_probs = Softmax(dim=1)(alphas)
        
        l = []
        
        #next_starting_node = edge_index_dest[alpha_probs.argmax(dim=1)]
        
        for i in range(alpha_probs.shape[1]):
            l.append((alpha_probs[0][i], edge_index_dest[i]))
        
        sorted(l, key=lambda x: x[0], reverse=True)
        
        # by default next_starting_node is set to starting_node
        next_starting_node = starting_node
        
        for i in range(len(l)):
            
            prob = l[i][0]
            
            # l[i][1] = node in linegraph (eg: could be node 12)
            # line_graph_nodes[l[i][1]] corresponds to node linegraph->primal graph (eg: node 12 refers to edge b/w nodes (4,5))
            # in primal graph
            node = line_graph_nodes[l[i][1]]
            u = node[0]
            v = node[1]
            
            if (int(u) not in node_set) or (int(v) not in node_set):
                next_starting_node = torch.tensor([l[i][1]])
                break
        
        return next_starting_node

