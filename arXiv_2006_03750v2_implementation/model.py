from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv
import torch
from torch.autograd import Variable
import math


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
    def __init__(self, hidden_dim, feature_dim):
        super(Decoder, self).__init__()
        
        self.const = 10
        
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        self.phi1 = Parameter(torch.randn(hidden_dim, feature_dim))
        self.phi2 = Parameter(torch.randn(hidden_dim, feature_dim))
        
    
    def forward(self, x, edge_index, node_set, line_graph_nodes, starting_node=None):
        
        #print(f"starting_node = {starting_node}")
        
        n_nodes = x.shape[0]
        n_edges = len(edge_index[0])

        alphas = (self.phi1 @ x.T).T @ (self.phi2 @ x.T)
        alphas = alphas / math.sqrt(self.hidden_dim)
        
        adj = torch.zeros(n_nodes, n_nodes)
        
        #by default next_node is starting_node (to avoid undecleration error if every node is already included in set) 
        next_starting_node = starting_node

        
        if starting_node != None:
            '''
            edge_index_sources = torch.where(edge_index[0] == starting_node)
            edge_index_dest = edge_index[1][edge_index_sources]
            for v in edge_index_dest:
                if int(v) not in node_set:
                    adj[starting_node.squeeze()][v] = 2.
                    adj[v][starting_node.squeeze()] = 2.
                else:
                    adj[starting_node.squeeze()][v] = 1.
                    adj[v][starting_node.squeeze()] = 1.
            '''
            edge_index_sources = torch.where(edge_index[0] == starting_node)
            edge_index_dest = edge_index[1][edge_index_sources]
            for v in edge_index_dest:
                adj[starting_node.squeeze()][v] = 1.
                adj[v][starting_node.squeeze()] = 1.
            
            z = torch.ones(n_nodes, n_nodes)
            for i in node_set:
                z[:,i] = 0
            
            probs = self.const * torch.tanh(alphas) + torch.tanh(adj + z)
            
            
            #taking softmax along the row ---->
            probs = Softmax(dim=1)(probs)
            #print(f"probs = {probs}")
            
            '''
            nodeIndexes_descOrd_ofprobs = torch.topk(probs[starting_node.squeeze()], n_nodes)[1] 
            for linegraphnode in nodeIndexes_descOrd_ofprobs:
                primal_edge = line_graph_nodes[linegraphnode]
                primal_node_u = int(primal_edge[0])
                primal_node_v = int(primal_edge[1])
                if (primal_node_u not in node_set) or (primal_node_v not in node_set):
                    next_starting_node = linegraphnode
                    print(f"next_starting_node = {next_starting_node}")
                    break
            '''
            
            next_starting_node = probs[starting_node.squeeze()].argmax(dim=0)
            
            #print(f"next_starting_node = {next_starting_node}")
            
        else:
            probs = self.const * torch.tanh(alphas) + adj
            #taking softmax along the row ---->
            probs = Softmax(dim=1)(probs)
            
            diagonal = torch.diagonal(probs)
            
            next_starting_node = diagonal.argmax(dim=0)
            #print(f"next_starting_node = {next_starting_node}")
        
        #add the already explored node in the set
        node_set.add(int(next_starting_node))
        print(f"dualGraph node_set = {node_set}")
        
        return next_starting_node, probs, node_set
