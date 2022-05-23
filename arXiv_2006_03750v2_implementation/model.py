from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv, SAGEConv
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical



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
        self.Wq = nn.Parameter(self.Wq)
        self.Wq.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))

        self.Wk = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wk = nn.Parameter(self.Wk)
        self.Wk.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))

        self.Wv = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wv = nn.Parameter(self.Wv)
        self.Wv.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))

        self.Wo = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wo = nn.Parameter(self.Wo)
        self.Wo.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))

        self.starting_node_W = torch.FloatTensor(self.hidden_dim)
        self.starting_node_W = nn.Parameter(self.starting_node_W)
        self.starting_node_W.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))

        self.previous_node_W = torch.FloatTensor(self.hidden_dim)
        self.previous_node_W = nn.Parameter(self.previous_node_W)
        self.previous_node_W.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))


    def forward(self, X, nodes_visited, starting_node=None, previous_node=None, x=None, adj=None, use_adj=False):

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

        if use_adj is True:
            if previous_node is None:
                mask2 = 0
            else:
                # mask2 ==> for adjacent nodes
                # mask2 = 1 only for adjacent nodes of the previous node
                mask2 = adj[previous_node]  ## <---------------------------------------- check it
                mask2 = mask2.view(1, n_nodes)
        else:
            mask2 = 0

        if self.final_layer == True:

            #[1 x n_nodes]
            out = self.C * torch.tanh(u) + mask + mask2

            out = out.view(-1)

            #[n_nodes]
            #uncomment the below
            #out = F.softmax(out, dim=0)
            #print(mask)
            #print(out)

        else:

            #[1 x n_nodes]
            u_ = F.softmax(u + mask + mask2, dim=1)

            #[1 x n_nodes].[n_nodes x hidden_dim] = [1 x hidden_dim]
            h = torch.matmul(u_, v)

            #[1 x hidden_dim].[hidden_dim x hidden_dim] = [1 x hidden_dim]
            out = torch.matmul(h, self.Wo)

            #[hidden_dim]
            out = out.view(-1)

        return out
        
        

    
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
        
        # forces the output corresponding to 1 node to lie between [0,1]
        # this is the prob for that node(of linegraph) to be included in the MST
        # note sigmoid is not acting on all nodes such that sum of probs for all nodes = 1. This is not the case
        # rather the only use of sigmoid is to make sure that output prob of node is between [0,1]
        x = self.sigmoid(x)
        
        
        #print(out)
        return x

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


class Encoder2_Layer(Module):
    def __init__(self, hidden_dim, isfirstLayer):
        super(Encoder2_Layer, self).__init__()

        self.isfirstLayer = isfirstLayer
        
        if isfirstLayer:
            self.gat = GAT(in_channels=1, hidden_channels=int(hidden_dim / 2), num_layers=3, out_channels=hidden_dim)
        else:
            self.gat = GAT(in_channels=hidden_dim, hidden_channels=int(hidden_dim / 2), num_layers=3, out_channels=hidden_dim)

        self.ReLU = ReLU(inplace=True)

        self.linear1 = Linear(in_features=1, out_features=hidden_dim)
        self.linear2 = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.linear3 = Linear(in_features=hidden_dim, out_features=hidden_dim)


    def forward(self, x, edge_index):

        '''
        x = [n_nodes]
        '''
        
        if self.isfirstLayer:
            #[n_nodes x hidden_dim]
            x1 = self.linear1(x)
        else:
            x1 = self.linear2(x)

        #[n_nodes x hidden_dim]
        x2 = self.gat(x, edge_index)

        #[n_nodes x hidden_dim]
        x2 = self.linear3(x2)

        #[n_nodes x hidden_dim]
        x2 = x1 + self.ReLU(x2)

        return x2


class Encoder2(Module):
    def __init__(self, hidden_dim):
        super(Encoder2, self).__init__()

        self.layer1 = Encoder2_Layer(hidden_dim=hidden_dim, isfirstLayer=True)
        self.layer2 = Encoder2_Layer(hidden_dim=hidden_dim, isfirstLayer=False)

        self.output_layer = Linear(in_features=hidden_dim, out_features=hidden_dim)

    def forward(self, x, edge_index):

        out = self.layer1(x, edge_index)
        out = self.layer2(out, edge_index)
        out = self.output_layer(out)

        # [n_nodes x hidden_dim]
        return out


class Decoder_Layer_StartingNode(Module):
    def __init__(self, hidden_dim):
        super(Decoder_Layer_StartingNode, self).__init__()

        self.hidden_dim = hidden_dim

        self.linear1 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.linear2 = Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2))
        self.linear3 = Linear(in_features=int(self.hidden_dim / 2), out_features=int(self.hidden_dim / 4))
        self.linear4 = Linear(in_features=int(self.hidden_dim / 4), out_features=1)

        self.sigmoid = Sigmoid()
        self.ReLU = ReLU(inplace=True)

    def forward(self, x):

        # x = [n_nodes x hidden_dim]

        out = self.linear1(x)
        out = self.ReLU(out)
        out = self.linear2(out)
        out = self.ReLU(out)
        out = self.linear3(out)
        out = self.ReLU(out)
        out = self.linear4(out)

        # prob of that node being a starting node = [1, n_nodes]
        out = self.sigmoid(out)

        out = out.view(-1)

        return out


class Decoder_Layer(Module):
    def __init__(self, hidden_dim, final_layer):
        super(Decoder_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.final_layer = final_layer

        self.x_concat_dim = 3
        
        self.C = 10

        self.Wq = torch.FloatTensor(self.x_concat_dim * self.hidden_dim, self.hidden_dim)
        self.Wq = nn.Parameter(self.Wq)
        self.Wq.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

        self.Wk = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wk = nn.Parameter(self.Wk)
        self.Wk.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

        self.Wv = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wv = nn.Parameter(self.Wv)
        self.Wv.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

        self.Wo = torch.FloatTensor(self.hidden_dim, self.hidden_dim)
        self.Wo = nn.Parameter(self.Wo)
        self.Wo.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

    def forward(self, x, X, starting_node, previous_node, nodes_visited, adj):

        '''
        x = [n_nodes x hidden_dim]
        X = None for first iteration else out of previous decoder layer ([hidden_dim])
        starting_node = the starting node for mst
        previous_node = the node output in the previous iteration
        '''

        n_nodes = x.shape[0]

        if X is None:
            # [3*hidden_dim]
            features_concat = torch.cat([x.mean(dim=0).squeeze(), x[starting_node], x[previous_node]])
        else:
            features_concat = torch.cat([X, x[starting_node], x[previous_node]])

        # [1 x 3*hidden_dim].[3*hidden_dim x hidden_dim] = [1 x hidden_dim]
        q = torch.matmul(features_concat, self.Wq)

        # [n_nodes x hidden_dim].[hidden_dim x hidden_dim] = [n_nodes x hidden_dim]
        k = torch.matmul(x, self.Wk)

        # [n_nodes x hidden_dim].[hidden_dim x hidden_dim] = [n_nodes x hidden_dim]
        v = torch.matmul(x, self.Wv)

        # [1 x hidden_dim].[hidden_dim x n_nodes] = [1 x n_nodes]
        u = torch.matmul(q, k.T) / math.sqrt(self.hidden_dim)

        # provide information regarding visited_nodes and the adjacent_nodes through masks

        # mask1 ==> already visited nodes
        # mask1 = 1 for all unvisited nodes
        mask1 = torch.ones(1, n_nodes)
        for node in nodes_visited:
            mask1[0][node] = 0

        # mask2 ==> for adjacent nodes
        # mask2 = 1 only for adjacent nodes of the previous node
        mask2 = adj[previous_node]      ## <---------------------------------------- check it
        mask2 = mask2.view(-1, n_nodes)

        if self.final_layer:

            # [1 x n_nodes]
            out = self.C * torch.tanh(u) + mask1 + mask2

            out = out.view(-1)

            # [n_nodes]
            out = F.softmax(out, dim=0)

        else:

            # [1 x n_nodes]
            u_ = F.softmax(u + mask1 + mask2, dim=1)

            # [1 x n_nodes].[n_nodes x hidden_dim] = [1 x hidden_dim]
            h = torch.matmul(u_, v)

            # [1 x hidden_dim].[hidden_dim x hidden_dim] = [1 x hidden_dim]
            out = torch.matmul(h, self.Wo)

            # [hidden_dim]
            out = out.view(-1)

        return out


class Decoder2(Module):
    def __init__(self, hidden_dim):
        super(Decoder2, self).__init__()

        self.decoder_layer1 = Decoder_Layer(hidden_dim, False)
        self.decoder_layer2 = Decoder_Layer(hidden_dim, True)

    def forward(self, x, starting_node, previous_node, nodes_visited, adj):

        out = self.decoder_layer1(x, None, starting_node, previous_node, nodes_visited, adj)
        out = self.decoder_layer2(x, out, starting_node, previous_node, nodes_visited, adj)

        return out


class ProbGen(Module):
    def __init__(self, hidden_dim):
        super(ProbGen, self).__init__()

        self.hidden_dim = hidden_dim

        self.linear1 = Linear(in_features=self.hidden_dim, out_features=int(self.hidden_dim / 2))
        self.linear2 = Linear(in_features=int(self.hidden_dim / 2), out_features=1)

        self.sigmoid = Sigmoid()
        self.ReLU = ReLU(inplace=True)

    def forward(self, x):

        '''
        x = encoded_features[current_node]
        '''

        out = self.linear1(x)
        out = self.ReLU(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

