from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv, SAGEConv
import torch
from torch.autograd import Variable
import math
import torch.nn.functional as F
from torch import nn
from dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from generate_graph import show_graph
from networkx.algorithms.tree.recognition import is_tree
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os


def create_graph_from_model_output(out, line_graph_nodes, graph):
    nodes_and_weights_list = []
    for i in range(len(out)):
        node_of_linegraph = out[i]
        u = line_graph_nodes[node_of_linegraph][0].numpy()
        v = line_graph_nodes[node_of_linegraph][1].numpy()
        nodes_and_weights_list.append((int(u), int(v), {'weight': int(graph.weight[node_of_linegraph])}))

    G = nx.Graph()
    G.add_edges_from(nodes_and_weights_list)

    return G


def calc_optimality(mst, G, n_nodes, mst_wt, calc_all=False):
    # by default optimality = -1
    optimality = -1

    sum_of_weights = 0
    for (_, _, w) in G.edges(data=True):
        sum_of_weights += w['weight']

    if not calc_all:

        # print(f"constructed mst total_wt = {sum_of_weights}")
        # print(f"actual mst total_wt= {mst_wt}")

        if is_tree(G) and G.number_of_nodes() == n_nodes:
            optimality = sum_of_weights / mst_wt

            # this was done to keep optimality always above 1.
            '''
            if optimality < 1.:
                optimality = 1 / optimality
            '''

            '''
            print("*********************************")
            print("The Generated MST is atleast a proper MST")
            print(f"optimality = {optimality}")
            print("*********************************")
            '''

    else:

        optimality = sum_of_weights / mst_wt

        # this was done to keep optimality always above 1.
        '''
        if optimality < 1.:
            optimality = 1 / optimality
        '''

    return optimality


def calc_reward(graph, n_nodes):
    # MAX_INDICATOR_VALUE = 1000
    # MIN_INDICATOR_VALUE = 0

    # indicator_function = 0 if is_tree(graph) else 1000
    # indicator_function = MIN_INDICATOR_VALUE if is_tree(graph) else MAX_INDICATOR_VALUE

    sum_of_weights = 0
    for (_, _, w) in graph.edges(data=True):
        sum_of_weights += w['weight']

    # reward = -torch.tensor(indicator_function + sum_of_weights, dtype=float ,requires_grad=requires_grad)
    # return reward
    return sum_of_weights


def add_nodes_in_graph(graph, u, v, wt):
    graph.add_edges_from([(u, v, {'weight': wt})])
    return graph


def save_model(model, name):
    current_directory = os.getcwd()
    filename = os.path.join(current_directory, name)
    torch.save(model.state_dict(), filename)


def check_early_stop(train_loss, epoch):
    early_stop_size = 10
    early_pause = False
    for i in range(epoch, epoch - early_stop_size, -1):
        if i >= 0 and i - 1 >= 0:
            if train_loss[i] < train_loss[i - 1]:
                early_pause = False
                break
            elif i == epoch - early_stop_size + 1:
                early_pause = True
        else:
            early_pause = False

    return early_pause


########################################################################################################################
########################################################################################################################


class Decoder(Module):
    def __init__(self, final_layer=False):
        super(Decoder, self).__init__()

        self.final_layer = final_layer

        self.const = 10

        self.hidden_dim = 256

        self.feature_dim = 1

        self.C = 10

        # dim of [x.mean(), x[starting_node], x[previous_node]]
        self.x_concat_dim = 3

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

        self.starting_node_W = torch.FloatTensor(self.hidden_dim)
        self.starting_node_W = nn.Parameter(self.starting_node_W)
        self.starting_node_W.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

        self.previous_node_W = torch.FloatTensor(self.hidden_dim)
        self.previous_node_W = nn.Parameter(self.previous_node_W)
        self.previous_node_W.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))

    def forward(self, X, nodes_visited, starting_node=None, previous_node=None, x=None, adj=None, use_adj=False):

        '''
        X = n_nodes x hidden_dim
        x = X.mean(dim=0) for 1st run, out of previous decoder for 2nd run => for both cases [hidden_dim]
        '''

        n_nodes = X.shape[0]

        if starting_node == None:

            # [hidden_dim]
            starting_node_features = self.starting_node_W
            previous_node_features = self.previous_node_W

        else:

            # [hidden_dim]
            starting_node_features = X[starting_node]
            previous_node_features = X[previous_node]

        # x = x.view(-1,)
        # [3*hidden_dim]
        features_concat = torch.cat([x, starting_node_features, previous_node_features])

        # [1 x 3*hidden_dim].[3*hidden_dim x hidden_dim] = [1 x hidden_dim]
        q = torch.matmul(features_concat, self.Wq)

        # [n_nodes x hidden_dim].[hidden_dim x hidden_dim] = [n_nodes x hidden_dim]
        k = torch.matmul(X, self.Wk)

        # [n_nodes x hidden_dim].[hidden_dim x hidden_dim] = [n_nodes x hidden_dim]
        v = torch.matmul(X, self.Wv)

        # [1 x hidden_dim].[hidden_dim x n_nodes] = [1 x n_nodes]
        u = torch.matmul(q, k.T) / math.sqrt(self.hidden_dim)

        # mask = 1 for all unvisited nodes
        mask = torch.ones(1, n_nodes)
        # mask = torch.zeros(1, n_nodes)
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

            # [1 x n_nodes]
            out = self.C * torch.tanh(u) + mask + mask2

            out = out.view(-1)

            # [n_nodes]
            # uncomment the below
            # out = F.softmax(out, dim=0)
            # print(mask)
            # print(out)

        else:

            # [1 x n_nodes]
            u_ = F.softmax(u + mask + mask2, dim=1)

            # [1 x n_nodes].[n_nodes x hidden_dim] = [1 x hidden_dim]
            h = torch.matmul(u_, v)

            # [1 x hidden_dim].[hidden_dim x hidden_dim] = [1 x hidden_dim]
            out = torch.matmul(h, self.Wo)

            # [hidden_dim]
            out = out.view(-1)

        return out


class Encoder(Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.hidden_dim = 256

        self.gat1 = GAT(in_channels=1, hidden_channels=int(self.hidden_dim / 2), num_layers=3,
                        out_channels=self.hidden_dim)
        self.gat2 = GAT(in_channels=self.hidden_dim, hidden_channels=self.hidden_dim, num_layers=3,
                        out_channels=self.hidden_dim)

        self.leakyReLU = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        self.linear1 = Linear(in_features=1, out_features=self.hidden_dim)
        self.linear2 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.linear3 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.linear4 = Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

    def forward(self, x, edge_index):
        '''
        x = [n_nodes]
        '''

        # [n_nodes x hidden_dim]
        x1 = self.linear1(x)

        # [n_nodes x hidden_dim]
        x2 = self.gat1(x, edge_index)

        # [n_nodes x hidden_dim]
        x2 = self.linear2(x2)

        # [n_nodes x hidden_dim]
        x2 = self.leakyReLU(x1 + x2)

        # [n_nodes x hidden_dim]
        x3 = self.gat2(x2, edge_index)

        # [n_nodes x hidden_dim]
        x3 = self.linear3(x3)

        # [n_nodes x hidden_dim]
        #out = self.leakyReLU(x1 + x2 + x3)

        #out = self.linear4(out)
        out = nn.Sigmoid()(x2 + x3)

        # print("encoder output")
        # print(out)
        return out


########################################################################################################################
########################################################################################################################

'''
Using Gumbel Softmax
'''

C1 = 1.25
C2 = 1.


def train(n_nodes, n_edges, use_adj):
    batch_size = 16

    epochs = 20
    train_data_len = 640
    val_data_len = 640

    e1 = 2  # after every e1 epochs print graphs
    e2 = 2  # after every e2 epoch save model to same file

    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    maxi = 1000000
    saved_epoch = -1

    encoder = Encoder().to(device)
    decoder1 = Decoder().to(device)
    decoder2 = Decoder(final_layer=True).to(device)

    tr_loader = GraphDataset(n_nodes, n_edges)
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer_decoder1 = torch.optim.SGD(decoder1.parameters(), lr=1e-3, weight_decay=0.01)
    optimizer_decoder2 = torch.optim.SGD(decoder2.parameters(), lr=1e-3, weight_decay=0.01)

    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=400, gamma=0.1)
    scheduler_decoder1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder1, step_size=400, gamma=0.1)
    scheduler_decoder2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder2, step_size=400, gamma=0.1)
    scheduler_decoder2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder2, step_size=400, gamma=0.1)

    train_loss_list = []
    val_loss_list = []

    optimality_val = []
    optimality_val_complete = []  # of all the generated mst's [doesn't matter whether proper or not]
    optimality_train = []

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    train_graph_list = []
    train_linegraph_list = []
    train_mst_list = []
    train_line_graph_nodes_list = []
    train_mst_wt_list = []
    train_target_entropy_list = []

    val_graph_list = []
    val_linegraph_list = []
    val_mst_list = []
    val_line_graph_nodes_list = []
    val_mst_wt_list = []
    val_target_entropy_list = []

    # store all the data elements in corresponding lists for training and val
    for d in range(train_data_len):
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):
            train_graph_list.append(graph)
            train_linegraph_list.append(linegraph)
            train_mst_list.append(mst)
            train_line_graph_nodes_list.append(line_graph_nodes)
            train_mst_wt_list.append(mst_wt)
            train_target_entropy_list.append(target_entropy)
    for d in range(val_data_len):
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):
            val_graph_list.append(graph)
            val_linegraph_list.append(linegraph)
            val_mst_list.append(mst)
            val_line_graph_nodes_list.append(line_graph_nodes)
            val_mst_wt_list.append(mst_wt)
            val_target_entropy_list.append(target_entropy)

    # 1 epoch means iterating over whole data
    for epoch in range(epochs):

        print(f"Epoch {epoch}...")

        # early_stop = False

        # TRAINING
        encoder.train()
        decoder1.train()
        decoder2.train()
        for cnt in range(int(train_data_len / batch_size)):

            pred_probs = []
            pred_mst_wt = []
            target_probs = []
            target_mst_wt = []

            # Clear gradients
            optimizer_encoder.zero_grad()
            optimizer_decoder1.zero_grad()
            optimizer_decoder2.zero_grad()

            for d in range(batch_size):
                dd = cnt * batch_size + d
                graph = train_graph_list[dd]
                linegraph = train_linegraph_list[dd]
                mst = train_mst_list[dd]
                line_graph_nodes = train_line_graph_nodes_list[dd]
                mst_wt = train_mst_wt_list[dd]
                target_entropy = train_target_entropy_list[dd]

                # linegraph.x[node] = edge wts of primal graph which are actually features in dual graph
                linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                encoded_features = encoder(linegraph.x, linegraph.edge_index)

                # create adj matrix from edge index of linegraph
                adj = torch.zeros(encoded_features.shape[0], encoded_features.shape[0])
                for i in range(len(linegraph.edge_index[0])):
                    u = linegraph.edge_index[0][i]
                    v = linegraph.edge_index[1][i]
                    adj[u][v] = 1
                    adj[v][u] = 1

                # create an empty graph
                G = nx.Graph()

                reward = torch.tensor([0.0], requires_grad=True)
                reward.grad = None

                # this set is for nodes in linegraph
                node_set = set()

                starting_node = None
                previous_node = None

                # mst_wt = torch.tensor(mst_wt, requires_grad=True)

                node_probs_entropy = torch.zeros(encoded_features.shape[0], requires_grad=True)
                node_probs_entropy.grad = None

                for i in range(n_nodes - 1):

                    if i == 0:
                        x = encoded_features.mean(dim=0).squeeze()
                    else:
                        x = out

                    out = decoder1(encoded_features, node_set, starting_node, previous_node, x, adj, use_adj)
                    logits = decoder2(encoded_features, node_set, starting_node, previous_node, out, adj, use_adj)

                    # generate samples over nodes [0,0,1,0,0,0...]
                    # at a time only 1 sample node will be 1 in one-hot vector
                    h = F.gumbel_softmax(logits, tau=1, hard=True)

                    # this should be fine as we are differentiating for node_probs_entropy which is summation over h and
                    # not over next node so argmax is actually not taken into account.
                    next_node = torch.argmax(h)

                    # ideally this should be of the form [1,0,0,1,1,0,1...]
                    # however adding ensures we get form:[2,0,0,0,1,0,1...] => this is undesirable, and therefore
                    # L1 loss should work fine on this
                    node_probs_entropy = node_probs_entropy + h

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node

                    if (starting_node == None):
                        starting_node = next_node

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    reward = reward + (linegraph.x[next_node])

                    # add the current node in the node set
                    node_set.add(int(next_node))

                pred_probs.append(node_probs_entropy)
                pred_mst_wt.append(reward)
                target_probs.append(target_entropy.squeeze())
                target_mst_wt.append(mst_wt)

                # optimality_train.append(calc_optimality(mst,G,n_nodes, mst_wt))

            pred_probs_tensor = torch.stack(pred_probs, dim=0)
            pred_mst_wt_tensor = torch.stack(pred_mst_wt, dim=0)
            target_probs_tensor = torch.stack(target_probs, dim=0)
            target_mst_wt_tensor = torch.stack(target_mst_wt, dim=0)

            loss = 10 * nn.L1Loss()(pred_probs_tensor, target_probs_tensor)
                   #nn.L1Loss()(pred_mst_wt_tensor, target_mst_wt_tensor) * \

            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            optimizer_decoder1.step()  # Update parameters based on gradients.
            optimizer_decoder2.step()  # Update parameters based on gradients.

            scheduler_encoder.step()
            scheduler_decoder1.step()
            scheduler_decoder2.step()

            train_loss_list.append(loss.item())

        # VALIDATION
        encoder.eval()
        decoder1.eval()
        decoder2.eval()
        for cnt in range(int(val_data_len / batch_size)):

            pred_probs = []
            pred_mst_wt = []
            target_probs = []
            target_mst_wt = []

            for d in range(batch_size):
                dd = cnt * batch_size + d
                graph = val_graph_list[dd]
                linegraph = val_linegraph_list[dd]
                mst = val_mst_list[dd]
                line_graph_nodes = val_line_graph_nodes_list[dd]
                mst_wt = val_mst_wt_list[dd]
                target_entropy = val_target_entropy_list[dd]

                linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

                reward = 0.0

                # this set is for nodes in linegraph
                node_set = set()

                with torch.no_grad():
                    # total number of nodes in primal graph
                    n_nodes = graph.num_nodes

                    # encoder returns probs of starting node
                    encoded_features = encoder(linegraph.x, linegraph.edge_index)

                    # create adj matrix from edge index of linegraph
                    adj = torch.zeros(encoded_features.shape[0], encoded_features.shape[0])
                    for i in range(len(linegraph.edge_index[0])):
                        u = linegraph.edge_index[0][i]
                        v = linegraph.edge_index[1][i]
                        adj[u][v] = 1
                        adj[v][u] = 1

                    # create an empty graph
                    G = nx.Graph()

                    starting_node = None
                    previous_node = None

                    # mst_wt = torch.tensor(mst_wt,requires_grad=True)

                    node_probs_entropy = torch.zeros(encoded_features.shape[0], requires_grad=True)

                    for i in range(n_nodes - 1):

                        if (i == 0):
                            x = encoded_features.mean(dim=0).squeeze()
                        else:
                            x = out

                        out = decoder1(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                        logits = decoder2(encoded_features, node_set, starting_node, previous_node, out, use_adj)

                        h = F.gumbel_softmax(logits, tau=1, hard=True)
                        node_probs_entropy = node_probs_entropy + h
                        next_node = torch.argmax(h)

                        # once we have next node we make it our starting node and begin the algorithm all over again
                        previous_node = next_node

                        if starting_node is None:
                            starting_node = next_node

                        # add edge between nodes in primal to constructed tree
                        u = int(line_graph_nodes[next_node][0])
                        v = int(line_graph_nodes[next_node][1])
                        G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                        reward += (linegraph.x[next_node])

                        # add the current node in the node set
                        node_set.add(int(next_node))

                pred_probs.append(node_probs_entropy)
                pred_mst_wt.append(reward)
                target_probs.append(target_entropy.squeeze())
                target_mst_wt.append(mst_wt)

                opt_val = calc_optimality(mst, G, n_nodes, mst_wt)
                optimality_val.append(opt_val)

                opt_val_comp = calc_optimality(mst, G, n_nodes, mst_wt, calc_all=True)
                optimality_val_complete.append(opt_val_comp)

                # '''
                if epoch % e1 == 0 and dd == 0:
                    print("VALIDATION:\n")
                    print("Original Graph:")
                    show_graph(to_networkx(graph, to_undirected=True))
                    print("LineGraph:")
                    show_graph(to_networkx(linegraph, to_undirected=True))
                    print("Output MST:")
                    show_graph(G)
                    print("Actual MST:")
                    show_graph(to_networkx(mst, to_undirected=True))
                    print("Loss:")
                    print(loss.item())
                    print("\nOptimality Tree:")
                    print(opt_val)
                    print("\nOptimality Complete (tree + notree):")
                    print(opt_val_comp)
                    print("")
                # '''

            pred_probs_tensor = torch.stack(pred_probs, dim=0)
            pred_mst_wt_tensor = torch.stack(pred_mst_wt, dim=0)
            target_probs_tensor = torch.stack(target_probs, dim=0)
            target_mst_wt_tensor = torch.stack(target_mst_wt, dim=0)

            loss = 10 * nn.L1Loss()(pred_probs_tensor, target_probs_tensor)
                   #(10 if is_tree(G) else 1) #* nn.L1Loss()(pred_mst_wt_tensor, target_mst_wt_tensor) * \

            val_loss_list.append(loss.item())

        # end training and val, now save model
        if epoch % e2 == 0:
            if epoch == 0:
                print(f"Model saved at epoch {epoch}")
                save_model(encoder, "mymodel_encoder.pth")
                save_model(decoder1, "mymodel_decoder1.pth")
                save_model(decoder2, "mymodel_decoder2.pth")
            else:
                if val_loss_list[epoch] <= maxi:
                    maxi = val_loss_list[epoch]
                    saved_epoch = epoch
                    print(f"Model saved at epoch {epoch}")
                    save_model(encoder, "mymodel_encoder.pth")
                    save_model(decoder1, "mymodel_decoder1.pth")
                    save_model(decoder2, "mymodel_decoder2.pth")

    # print(f"optimality_val = {optimality_val}")

    optimality_avg = torch.tensor(optimality_val).float()
    optimality_avg_complete = torch.tensor(optimality_val_complete).float()

    print("\n#####################################################")
    print("Final Report")
    print(f"Total Number of Different Graphs = {epochs}")
    print(f"Total Number of MST's generated by model = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print(f"Avg Optimality of all generated MST's (proper/improper) = {optimality_avg_complete.mean(dim=0)}")
    print(f"Model saved at epoch={saved_epoch}")
    print("#####################################################")

    return train_loss_list, val_loss_list


# TEST
def test(n_nodes, n_edges, use_adj):
    batch_size = 1
    epochs = 100
    e = 1
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    encoder = Encoder().to(device)
    decoder1 = Decoder().to(device)
    decoder2 = Decoder(final_layer=True).to(device)

    PATH1 = "mymodel_encoder.pth"
    PATH2 = "mymodel_decoder1.pth"
    PATH3 = "mymodel_decoder2.pth"

    encoder.load_state_dict(torch.load(PATH1))
    decoder1.load_state_dict(torch.load(PATH2))
    decoder2.load_state_dict(torch.load(PATH3))

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    te_loader = GraphDataset(n_nodes, n_edges)
    test_loader = DataLoader(te_loader, batch_size=batch_size)

    train_loss = []
    val_loss = []

    optimality_val = []
    optimality_val_complete = []  # of all the generated mst's [doesn't matter whether proper or not]
    optimality_train = []

    PENALTY = 1000

    for epoch in range(epochs):

        if epoch % e == 0:
            print(f"Epoch {epoch}...")

        # TESTING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(test_loader):

            encoder.eval()
            decoder1.eval()
            decoder2.eval()

            linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

            reward = 0

            # this set is for nodes in linegraph
            node_set = set()

            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                encoded_features = encoder(linegraph.x, linegraph.edge_index)

                # create adj matrix from edge index of linegraph
                adj = torch.zeros(encoded_features.shape[0], encoded_features.shape[0])
                for i in range(len(linegraph.edge_index[0])):
                    u = linegraph.edge_index[0][i]
                    v = linegraph.edge_index[1][i]
                    adj[u][v] = 1
                    adj[v][u] = 1

                # create an empty graph
                G = nx.Graph()

                starting_node = None
                previous_node = None

                node_probs_entropy = torch.zeros(encoded_features.shape[0])

                # mst_wt = torch.tensor(mst_wt,requires_grad=True)

                for i in range(n_nodes - 1):

                    if (i == 0):
                        x = encoded_features.mean(dim=0).squeeze()
                    else:
                        x = out

                    out = decoder1(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    logits = decoder2(encoded_features, node_set, starting_node, previous_node, out, use_adj)

                    h = F.gumbel_softmax(logits, tau=1, hard=True)
                    node_probs_entropy += h
                    next_node = torch.argmax(h)

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node

                    if starting_node is None:
                        starting_node = next_node

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    reward += (linegraph.x[next_node])

                    # add the current node in the node set
                    node_set.add(int(next_node))

            loss = 10 * nn.L1Loss()(node_probs_entropy, target_entropy.squeeze())# * nn.L1Loss()(reward, mst_wt)

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))
            optimality_val_complete.append(calc_optimality(mst, G, n_nodes, mst_wt, calc_all=True))

            val_loss.append(loss.item())
            # '''
            if epoch % e == 0:
                print("Original Graph:")
                show_graph(to_networkx(graph, to_undirected=True))
                print("LineGraph:")
                show_graph(to_networkx(linegraph, to_undirected=True))
                print("Output MST:")
                show_graph(G)
                print("Actual MST:")
                show_graph(to_networkx(mst, to_undirected=True))
                print("Loss:")
                print(loss.item())
                print("\nOptimality:")
                print(optimality_val_complete[epoch])
                print("")
            # '''

    optimality_avg = torch.tensor(optimality_val).float()
    optimality_avg_complete = torch.tensor(optimality_val_complete).float()

    print("\n#####################################################")
    print("Final Report")
    print(f"Total Number of Randomly Generated Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print(f"Avg Optimality of all generated MST's (proper/improper) = {optimality_avg_complete.mean(dim=0)}")
    print("#####################################################")
