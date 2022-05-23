from generate_graph import create_graph_dataset
import torch
from model import Encoder, Decoder, Decoder_Entropy, Encoder2, Decoder_Layer_StartingNode, Decoder2, ProbGen
from dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from generate_graph import show_graph
from networkx.algorithms.tree.recognition import is_tree
from torch import nn
import torch.nn.functional as F
from networkx.classes.function import number_of_nodes
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.algorithms.similarity import graph_edit_distance
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
    
    #by default optimality = -1
    optimality = -1
    
    sum_of_weights = 0
    for (_, _, w) in G.edges(data=True):
        sum_of_weights += w['weight']
    
    if not calc_all:

        #print(f"constructed mst total_wt = {sum_of_weights}")
        #print(f"actual mst total_wt= {mst_wt}")

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


def loss_func(reward, prob, graph, n_nodes, penalty, mst_wt, C):
    
    TREE_PENALTY = 10
    N_NODES_PENALTY = 100
    
    #beta = 0.8

    #loss = -torch.mean(prob)
    
    #loss = -torch.mean(reward * prob)
    
    #t is always greater than or equal to 1
    t = torch.tensor(n_nodes - graph.number_of_nodes())
    
    l1 = nn.L1Loss()
    
    #loss = torch.abs(torch.exp(reward) - torch.exp(mst_wt))
    
    #loss = torch.abs(torch.log(reward) - torch.log(mst_wt))
    
    
    loss = l1(mst_wt, reward)
    
    #loss += (t * t)
    
    loss += torch.tensor(2).pow(t)
    
    #loss = torch.tensor(2).pow(loss)
    
    #print(f"loss={loss}")
    #loss = loss * (1. - (100. * torch.log(t)))
    #print(f"loss after mult ={loss}")
    
    #print(f"is_tree = {is_tree(graph)}")
    #print(f"graph.number_of_nodes() = {graph.number_of_nodes()}, n_nodes = {n_nodes}")
    
    #loss = loss * (1 if is_tree(graph) else TREE_PENALTY)
    #loss = loss * (1 if graph.number_of_nodes() == n_nodes else N_NODES_PENALTY)
    
    #loss = loss + (0 if is_tree(graph) else 1000)
    
    #loss *= penalty
    #loss += (0 if graph.number_of_nodes() == n_nodes else 1000)
    
    '''
    logprobs = prob
    
    if(C == -1):
        C = reward
    else:
        C = (C * beta) + ((1. - beta)*reward)
    
    loss = ((reward - C) * logprobs).mean()
    '''
    
    return loss, C
    


def calc_reward(graph, n_nodes):
    
    #MAX_INDICATOR_VALUE = 1000
    #MIN_INDICATOR_VALUE = 0
    
    # indicator_function = 0 if is_tree(graph) else 1000
    #indicator_function = MIN_INDICATOR_VALUE if is_tree(graph) else MAX_INDICATOR_VALUE
    
    sum_of_weights = 0
    for (_, _, w) in graph.edges(data=True):
        sum_of_weights += w['weight']
    
    
    #reward = -torch.tensor(indicator_function + sum_of_weights, dtype=float ,requires_grad=requires_grad)
    #return reward
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
        if i >= 0 and i-1 >= 0:
            if train_loss[i] < train_loss[i-1]:
                early_pause = False
                break
            elif i == epoch - early_stop_size + 1:
                early_pause = True
        else:
            early_pause = False
    
    return early_pause
            
########################################################################################################################
########################################################################################################################

"""
C1 = 1.
C2 = 1.


def train(n_nodes, n_edges, use_adj):
    batch_size = 1

    epochs = 1000

    e1 = 100  # after every e1 epochs print graphs
    e2 = 20  # after every e2 epoch save model to same file

    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    maxi = 1000
    saved_epoch = -1
    
    encoder = Encoder().to(device)
    decoder1 = Decoder().to(device)
    decoder2 = Decoder(final_layer=True).to(device)

    tr_loader = GraphDataset(n_nodes, n_edges)
    train_loader = DataLoader(tr_loader, batch_size=batch_size)
    
    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=1e-5)
    optimizer_decoder1 = torch.optim.SGD(decoder1.parameters(), lr=1e-5)
    optimizer_decoder2 = torch.optim.SGD(decoder2.parameters(), lr=1e-5)
    
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=250, gamma=0.1)
    scheduler_decoder1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder1, step_size=250, gamma=0.1)
    scheduler_decoder2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder2, step_size=2500, gamma=0.1)
    
    train_loss = []
    val_loss = []
    
    optimality_val = []
    optimality_val_complete = [] # of all the generated mst's [doesn't matter whether proper or not]
    optimality_train = []
    
    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    for epoch in range(epochs):
        
        if epoch % e1 == 0:
            print(f"Epoch {epoch}...")
        
        early_stop = False
        
        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):
            
            encoder.train()
            decoder1.train()
            decoder2.train()
            
            optimizer_encoder.zero_grad()# Clear gradients.
            optimizer_decoder1.zero_grad()# Clear gradients.
            optimizer_decoder2.zero_grad()# Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
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
            
            logprobs = 0
            reward = 0
            
            #this set is for nodes in linegraph
            node_set = set()
            
            starting_node = None
            previous_node = None
            
            #mst_wt = torch.tensor(mst_wt, requires_grad=True)
            
            node_probs_entropy = torch.zeros(encoded_features.shape[0])
            
            for i in range(n_nodes-1):
                
                if(i == 0):
                    x = encoded_features.mean(dim=0).squeeze()
                else:
                    x = out
                
                out = decoder1(encoded_features, node_set, starting_node, previous_node, x, adj, use_adj)
                probs = decoder2(encoded_features, node_set, starting_node, previous_node, x, adj, use_adj)
                
                # sample next node from probs
                #sampler = Categorical(probs)
                #next_node = sampler.sample()
                next_node = torch.argmax(probs, dim=0)
                
                node_probs_entropy[next_node] = probs[next_node]
                
                # once we have next node we make it our starting node and begin the algorithm all over again
                previous_node = next_node
                
                if(starting_node == None):
                    starting_node = next_node
                
                # add edge between nodes in primal to constructed tree
                u = int(line_graph_nodes[next_node][0])
                v = int(line_graph_nodes[next_node][1])
                G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])
                
                for node in range(probs.shape[0]):
                    reward += (probs[node] * linegraph.x[node])
                
                #add the current node in the node set
                node_set.add(int(next_node))
                
            #loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C1 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C2 * criterion2(reward, mst_wt)
            
            #optimality_train.append(calc_optimality(mst,G,n_nodes, mst_wt))
            
            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            optimizer_decoder1.step()  # Update parameters based on gradients.
            optimizer_decoder2.step()  # Update parameters based on gradients.
            
            #scheduler_encoder.step()
            #scheduler_decoder1.step()
            #scheduler_decoder2.step()
            
            train_loss.append(loss.item())
            
            #early_stop = check_early_stop(train_loss, epoch)
            
            
        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):
            
            encoder.eval()
            decoder1.eval()
            decoder2.eval()
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            logprobs = 0
            reward = 0
            
            #this set is for nodes in linegraph
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
                
                #mst_wt = torch.tensor(mst_wt,requires_grad=True)
                
                node_probs_entropy = torch.zeros(encoded_features.shape[0])
                
                for i in range(n_nodes-1):
                    
                    if(i == 0):
                        x = encoded_features.mean(dim=0).squeeze()
                    else:
                        x = out
                    
                    out = decoder1(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    probs = decoder2(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    
                    # sample next node from probs
                    #sampler = Categorical(probs)
                    #next_node = sampler.sample()
                    #print(probs)
                    next_node = torch.argmax(probs, dim=0)
                    
                    node_probs_entropy[next_node] = probs[next_node]

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node
                    
                    if(starting_node == None):
                        starting_node = next_node
                    
                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    for node in range(probs.shape[0]):
                        reward += (probs[node] * linegraph.x[node])
                    
                    #add the current node in the node set
                    node_set.add(int(next_node))
                
            #loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C1 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C2 * criterion2(reward, mst_wt)
            
            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))
            optimality_val_complete.append(calc_optimality(mst, G, n_nodes, mst_wt, calc_all=True))
            
            val_loss.append(loss.item())
            #'''
            if epoch % e1 == 0:
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
            #'''

        '''        
        if early_stop:
            print("\nStopping Ealry !!\n")
            break
        '''
        if epoch % e2 == 0:
            if epoch == 0:
                print(f"Model saved at epoch {epoch}")
                save_model(encoder, "mymodel_encoder.pth")
                save_model(decoder1, "mymodel_decoder1.pth")
                save_model(decoder2, "mymodel_decoder2.pth")
            else:
                if val_loss[epoch] <= maxi:
                    maxi = val_loss[epoch]
                    saved_epoch = epoch
                    print(f"Model saved at epoch {epoch}")
                    save_model(encoder, "mymodel_encoder.pth")
                    save_model(decoder1, "mymodel_decoder1.pth")
                    save_model(decoder2, "mymodel_decoder2.pth")


    #print(f"optimality_val = {optimality_val}")
    
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
    
    return train_loss, val_loss


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
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            logprobs = 0
            reward = 0
            
            #this set is for nodes in linegraph
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
                
                #mst_wt = torch.tensor(mst_wt,requires_grad=True)
                
                for i in range(n_nodes-1):
                    
                    if(i == 0):
                        x = encoded_features.mean(dim=0).squeeze()
                    else:
                        x = out
                    
                    out = decoder1(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    probs = decoder2(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    
                    # sample next node from probs
                    #sampler = Categorical(probs)
                    #next_node = sampler.sample()
                    #print(probs)
                    next_node = torch.argmax(probs, dim=0)
                    
                    node_probs_entropy[next_node] = probs[next_node]

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node
                    
                    if(starting_node == None):
                        starting_node = next_node
                    
                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    for node in range(probs.shape[0]):
                        reward += (probs[node] * linegraph.x[node])
                    
                    #add the current node in the node set
                    node_set.add(int(next_node))
                
            #loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C1 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C2 * criterion2(reward, mst_wt)
            
            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))
            optimality_val_complete.append(calc_optimality(mst, G, n_nodes, mst_wt, calc_all=True))
            
            val_loss.append(loss.item())
            #'''
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
            #'''
                
        
        '''
        if epoch % e == 0:
            save_model(encoder)
        '''
    #print(f"optimality_val = {optimality_val}")
    
    optimality_avg = torch.tensor(optimality_val).float()
    optimality_avg_complete = torch.tensor(optimality_val_complete).float()
    
    print("\n#####################################################")
    print("Final Report")
    print(f"Total Number of Randomly Generated Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print(f"Avg Optimality of all generated MST's (proper/improper) = {optimality_avg_complete.mean(dim=0)}")
    print("#####################################################")
    
"""

#################################################################################################################################


C21 = 0.4
C22 = 0.6

def train_entropy():
    batch_size = 1
    epochs = 2000
    e = 20
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    encoder = Encoder().to(device)
    decoder = Decoder_Entropy().to(device)

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=0.01)
    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=0.01)

    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=200, gamma=0.1)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=200, gamma=0.1)

    train_loss = []
    val_loss = []

    optimality_val = []
    optimality_train = []

    PENALTY = 10

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    early_stop = False

    for epoch in range(epochs):

        if epoch % e == 0:
            print(f"Epoch {epoch}...")

        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):

            encoder.train()
            decoder.train()

            optimizer_encoder.zero_grad()# Clear gradients.
            optimizer_decoder.zero_grad()# Clear gradients.

            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)

            # total number of nodes in primal graph
            n_nodes = graph.num_nodes

            # create an empty graph
            G = nx.Graph()

            #mst_wt = torch.tensor(mst_wt, requires_grad=True)
            #mst_wt = torch.tensor(mst_wt, requires_grad=True)

            ## NN
            encoded_features = encoder(linegraph.x, linegraph.edge_index)
            probs = decoder(encoded_features).squeeze().view(-1)
            #print(f"probs={probs}")

            values, indices = probs.topk(n_nodes-1)

            reward = 0

            for i in range(n_nodes-1):
                node = indices[i]

                reward += (linegraph.x[node] * probs[node])

                # add edge between nodes in primal to constructed tree
                u = int(line_graph_nodes[node][0])
                v = int(line_graph_nodes[node][1])
                G = add_nodes_in_graph(G, u, v, linegraph.x[node])

            loss = C21 * criterion1(probs, target_entropy.squeeze()) + C22 * criterion2(reward, mst_wt)

            '''
            if is_tree(G):
                loss *= PENALTY
            '''

            optimality_train.append(calc_optimality(mst, G, n_nodes, mst_wt))

            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            optimizer_decoder.step()  # Update parameters based on gradients.

            scheduler_encoder.step()
            scheduler_decoder.step()

            train_loss.append(loss.item())

            early_stop = check_early_stop(train_loss, epoch)



        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):

            encoder.eval()
            decoder.eval()

            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)

            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # create an empty graph
                G = nx.Graph()

                #mst_wt = torch.tensor(mst_wt, requires_grad=True)

                ## NN
                encoded_features = encoder(linegraph.x, linegraph.edge_index)
                probs = decoder(encoded_features).squeeze().view(-1)

                values, indices = probs.topk(n_nodes-1)

                reward = 0

                for i in range(n_nodes-1):
                    node = indices[i]

                    reward += (linegraph.x[node] * probs[node])

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[node][0])
                    v = int(line_graph_nodes[node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[node])

            loss = C21 * criterion1(probs, target_entropy.squeeze()) + C22 * criterion2(reward, mst_wt)

            '''
            if is_tree(G):
                loss *= PENALTY
            '''

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))

            val_loss.append(loss.item())
            #'''
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
                print("")
            #'''

        if early_stop:
            print("EARLY STOPPING !!")
            break
        '''
        if epoch % e == 0:
            save_model(encoder, "mymodel.pth")
        '''
    #print(f"optimality_val = {optimality_val}")

    optimality_avg = torch.tensor(optimality_val).float()

    print("#####################################################")
    print("Final Report: Validation Set")
    print(f"Total Number of Different Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print("#####################################################")

    save_model(encoder, "mymodel_encoder_entropy.pth")
    save_model(decoder, "mymodel_decoder_entropy.pth")

    return train_loss, val_loss


# TEST
def test_entropy():
    batch_size = 1
    epochs = 100
    e = 1
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    encoder = Encoder().to(device)
    decoder = Decoder_Entropy().to(device)

    PATH1 = "mymodel_encoder_entropy.pth"
    PATH2 = "mymodel_decoder_entropy.pth"

    encoder.load_state_dict(torch.load(PATH1))
    decoder.load_state_dict(torch.load(PATH2))


    te_loader = GraphDataset()
    test_loader = DataLoader(te_loader, batch_size=batch_size)

    train_loss = []
    val_loss = []

    optimality_val = []
    optimality_train = []

    PENALTY = 10

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    for epoch in range(epochs):

        if epoch % e == 0:
            print(f"Epoch {epoch}...")

        # TESTING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(test_loader):

            encoder.eval()
            decoder.eval()

            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)

            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # create an empty graph
                G = nx.Graph()

                #mst_wt = torch.tensor(mst_wt, requires_grad=True)

                ## NN
                encoded_features = encoder(linegraph.x, linegraph.edge_index)
                probs = decoder(encoded_features).squeeze().view(-1)

                print("How sure is the network:")
                print(probs)

                values, indices = probs.topk(n_nodes-1)

                reward = 0

                for i in range(n_nodes-1):
                    node = indices[i]

                    reward += (linegraph.x[node] * probs[node])

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[node][0])
                    v = int(line_graph_nodes[node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[node])

            loss = C21 * criterion1(probs, target_entropy.squeeze()) + C22 * criterion2(reward, mst_wt)

            '''
            if is_tree(G):
                loss *= PENALTY
            '''

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))

            val_loss.append(loss.item())
            #'''
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
                print("")
            #'''

    optimality_avg = torch.tensor(optimality_val).float()

    print("#####################################################")
    print("Final Report")
    print(f"Total Number of Different Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print("#####################################################")
    

########################################################################################################################
########################################################################################################################
########################################################################################################################

C31 = 1.
C32 = 0.


"""def train_model3():
    batch_size = 1
    epochs = 400
    e = 25
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    encoder = Encoder2(hidden_dim=128).to(device)
    decoder_starting_layer = Decoder_Layer_StartingNode(hidden_dim=128).to(device)
    decoder = Decoder2(hidden_dim=128).to(device)
    probgen = ProbGen(hidden_dim=128).to(device)

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=0.0001)
    optimizer_decoder_starting_layer = torch.optim.SGD(decoder_starting_layer.parameters(), lr=0.0001)
    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=0.0001)
    optimizer_probgen = torch.optim.SGD(probgen.parameters(), lr=0.0001)

    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=100, gamma=0.1)
    scheduler_decoder_starting_layer = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder_starting_layer, step_size=100, gamma=0.1)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder, step_size=100, gamma=0.1)
    scheduler_probgen = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_probgen, step_size=100, gamma=0.1)

    train_loss = []
    val_loss = []

    optimality_val = []
    optimality_train = []

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    for epoch in range(epochs):

        if epoch % e == 0:
            print(f"Epoch {epoch}...")

        early_stop = False

        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):

            encoder.train()
            decoder_starting_layer.train()
            decoder.train()

            optimizer_encoder.zero_grad()  # Clear gradients.
            optimizer_decoder_starting_layer.zero_grad()  # Clear gradients.
            optimizer_decoder.zero_grad()  # Clear gradients.

            linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

            # total number of nodes in primal graph
            n_nodes = graph.num_nodes

            # encoder returns probs of starting node [n_nodes x hidden_dim]
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

            reward = 0

            # this set is for nodes in linegraph
            node_set = set()

            starting_node = decoder_starting_layer(encoded_features)
            starting_node = torch.argmax(starting_node, dim=0)
            previous_node = starting_node

            node_probs_entropy = torch.zeros(encoded_features.shape[0])

            for i in range(n_nodes - 1):

                probs = decoder(encoded_features, starting_node, previous_node, node_set, adj)

                # sample next node from probs
                # sampler = Categorical(probs)
                # next_node = sampler.sample()
                next_node = torch.argmax(probs, dim=0)

                node_probs_entropy[next_node] = probs[next_node]

                # once we have next node we make it our starting node and begin the algorithm all over again
                previous_node = next_node

                # add edge between nodes in primal to constructed tree
                u = int(line_graph_nodes[next_node][0])
                v = int(line_graph_nodes[next_node][1])
                G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                for node in range(probs.shape[0]):
                    reward += (probs[node] * linegraph.x[node])

                # add the current node in the node set
                node_set.add(int(next_node))


            # loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C31 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C32 * criterion2(reward, mst_wt)

            optimality_train.append(calc_optimality(mst, G, n_nodes, mst_wt))

            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            optimizer_decoder_starting_layer.step()  # Update parameters based on gradients.
            optimizer_decoder.step()  # Update parameters based on gradients.

            scheduler_encoder.step()
            scheduler_decoder_starting_layer.step()
            scheduler_decoder.step()

            train_loss.append(loss.item())

            early_stop = check_early_stop(train_loss, epoch)

        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):

            encoder.eval()
            decoder_starting_layer.eval()
            decoder.eval()

            linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

            logprobs = 0
            reward = 0

            # this set is for nodes in linegraph
            node_set = set()

            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                encoded_features = encoder(linegraph.x, linegraph.edge_index)

                # create an empty graph
                G = nx.Graph()

                starting_node = decoder_starting_layer(encoded_features)
                starting_node = torch.argmax(starting_node, dim=0)
                previous_node = starting_node

                # create adj matrix from edge index of linegraph
                adj = torch.zeros(encoded_features.shape[0], encoded_features.shape[0])
                for i in range(len(linegraph.edge_index[0])):
                    u = linegraph.edge_index[0][i]
                    v = linegraph.edge_index[1][i]
                    adj[u][v] = 1
                    adj[v][u] = 1

                node_probs_entropy = torch.zeros(encoded_features.shape[0])

                for i in range(n_nodes - 1):

                    probs = decoder(encoded_features, starting_node, previous_node, node_set, adj)

                    # sample next node from probs
                    # sampler = Categorical(probs)
                    # next_node = sampler.sample()
                    # print(probs)
                    next_node = torch.argmax(probs, dim=0)

                    node_probs_entropy[next_node] = probs[next_node]

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    for node in range(probs.shape[0]):
                        reward += (probs[node] * linegraph.x[node])

                    # add the current node in the node set
                    node_set.add(int(next_node))

            # loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C31 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C32 * criterion2(reward, mst_wt)

            print("node_probs_entropy:")
            print(node_probs_entropy)
            print("target_entropy:")
            print(target_entropy.squeeze())
            
            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))

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
                print("")
            # '''

        if early_stop:
            print("\nStopping Ealry !!\n")
            break
        '''
        if epoch % e == 0:
            save_model(encoder, "mymodel.pth")
        '''
    # print(f"optimality_val = {optimality_val}")

    optimality_avg = torch.tensor(optimality_val).float()

    print("#####################################################")
    print("Final Report")
    print(f"Total Number of Different Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print("#####################################################")

    '''
    save_model(encoder, "mymodel_encoder.pth")
    save_model(decoder1, "mymodel_decoder1.pth")
    save_model(decoder2, "mymodel_decoder2.pth")
    '''

    return train_loss, val_loss
"""


def train_model3():
    batch_size = 1
    epochs = 1000
    e = 25
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'
    clamp_value = 0.35

    encoder = Encoder2(hidden_dim=128).to(device)
    decoder_starting_layer = Decoder_Layer_StartingNode(hidden_dim=128).to(device)
    decoder = Decoder2(hidden_dim=128).to(device)
    probgen = ProbGen(hidden_dim=128).to(device)

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=0.01)
    optimizer_decoder_starting_layer = torch.optim.SGD(decoder_starting_layer.parameters(), lr=0.01)
    optimizer_decoder = torch.optim.SGD(decoder.parameters(), lr=0.01)
    optimizer_probgen = torch.optim.SGD(probgen.parameters(), lr=0.01)

    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=100, gamma=0.1)
    scheduler_decoder_starting_layer = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder_starting_layer,
                                                                       step_size=100, gamma=0.1)
    scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder, step_size=100, gamma=0.1)
    scheduler_probgen = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_probgen, step_size=100, gamma=0.1)

    train_loss = []
    val_loss = []

    optimality_val = []
    optimality_train = []

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    for epoch in range(epochs):

        if epoch % e == 0:
            print(f"Epoch {epoch}...")

        early_stop = False

        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):

            encoder.train()
            decoder_starting_layer.train()
            decoder.train()

            optimizer_encoder.zero_grad()  # Clear gradients.
            optimizer_decoder_starting_layer.zero_grad()  # Clear gradients.
            optimizer_decoder.zero_grad()  # Clear gradients.

            linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

            # total number of nodes in primal graph
            n_nodes = graph.num_nodes

            # to check errors while backward()
            with torch.autograd.set_detect_anomaly(True):

                # encoder returns probs of starting node [n_nodes x hidden_dim]
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

                reward = 0

                # this set is for nodes in linegraph
                node_set = set()

                starting_node_probs = decoder_starting_layer(encoded_features)
                starting_node = torch.argmax(starting_node_probs, dim=0)
                prob_previous = starting_node_probs[starting_node].clone()
                previous_node = starting_node
                node_set.add(int(starting_node))

                node_probs_entropy = torch.zeros(encoded_features.shape[0])

                for i in range(n_nodes - 1):

                    probs = decoder(encoded_features, starting_node, previous_node, node_set, adj)

                    # sample next node from probs
                    # sampler = Categorical(probs)
                    # next_node = sampler.sample()
                    next_node = torch.argmax(probs, dim=0)

                    #print(node_set)

                    node_set.remove(int(previous_node))
                    node_set.add(int(next_node))
                    probs_cond = decoder(encoded_features, starting_node, next_node, node_set, adj)
                    node_set.remove(int(next_node))
                    node_set.add(int(previous_node))

                    node_probs_entropy[next_node] = probs[next_node] * prob_previous / (probs_cond[previous_node] + eps)
                    #print(node_probs_entropy[next_node])
                    #print(probs[next_node])
                    #print(prob_previous)
                    #print(probs_cond[previous_node])

                    if node_probs_entropy[next_node] > 1.:
                        node_probs_entropy[next_node] = 1.

                    prob_previous = node_probs_entropy[next_node].clone()

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    for node in range(probs.shape[0]):
                        reward += (probs[node] * linegraph.x[node])

                    # add the current node in the node set
                    node_set.add(int(next_node))

                print("node_probs_entropy:")
                print(node_probs_entropy)
                print("target_entropy:")
                print(target_entropy.squeeze())

                # loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
                loss = C31 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C32 * criterion2(reward, mst_wt)

                optimality_train.append(calc_optimality(mst, G, n_nodes, mst_wt))

                loss.backward()  # Derive gradients.
                optimizer_encoder.step()  # Update parameters based on gradients.
                optimizer_decoder_starting_layer.step()  # Update parameters based on gradients.
                optimizer_decoder.step()  # Update parameters based on gradients.

                scheduler_encoder.step()
                scheduler_decoder_starting_layer.step()
                scheduler_decoder.step()

                train_loss.append(loss.item())

                early_stop = check_early_stop(train_loss, epoch)

        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):

            encoder.eval()
            decoder_starting_layer.eval()
            decoder.eval()

            linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

            logprobs = 0
            reward = 0

            # this set is for nodes in linegraph
            node_set = set()

            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                encoded_features = encoder(linegraph.x, linegraph.edge_index)

                # create an empty graph
                G = nx.Graph()

                starting_node_probs = decoder_starting_layer(encoded_features)
                starting_node = torch.argmax(starting_node_probs, dim=0)
                prob_previous = starting_node_probs[starting_node].clone()
                previous_node = starting_node
                node_set.add(int(starting_node))

                # create adj matrix from edge index of linegraph
                adj = torch.zeros(encoded_features.shape[0], encoded_features.shape[0])
                for i in range(len(linegraph.edge_index[0])):
                    u = linegraph.edge_index[0][i]
                    v = linegraph.edge_index[1][i]
                    adj[u][v] = 1
                    adj[v][u] = 1

                node_probs_entropy = torch.zeros(encoded_features.shape[0])

                for i in range(n_nodes - 1):

                    probs = decoder(encoded_features, starting_node, previous_node, node_set, adj)

                    # sample next node from probs
                    # sampler = Categorical(probs)
                    # next_node = sampler.sample()
                    # print(probs)
                    next_node = torch.argmax(probs, dim=0)

                    node_set.remove(int(previous_node))
                    node_set.add(int(next_node))
                    probs_cond = decoder(encoded_features, starting_node, next_node, node_set, adj)
                    node_set.remove(int(next_node))
                    node_set.add(int(previous_node))

                    node_probs_entropy[next_node] = probs[next_node] * prob_previous / (probs_cond[previous_node] + eps)

                    if node_probs_entropy[next_node] > 1.:
                        node_probs_entropy[next_node] = 1.

                    prob_previous = node_probs_entropy[next_node].clone()

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    for node in range(probs.shape[0]):
                        reward += (probs[node] * linegraph.x[node])

                    # add the current node in the node set
                    node_set.add(int(next_node))

            print("node_probs_entropy:")
            print(node_probs_entropy)
            print("target_entropy:")
            print(target_entropy.squeeze())

            # loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C31 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C32 * criterion2(reward, mst_wt)

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))

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
                print("")
            # '''

        if early_stop:
            print("\nStopping Ealry !!\n")
            break
        '''
        if epoch % e == 0:
            save_model(encoder, "mymodel.pth")
        '''
    # print(f"optimality_val = {optimality_val}")

    optimality_avg = torch.tensor(optimality_val).float()

    print("#####################################################")
    print("Final Report")
    print(f"Total Number of Different Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print("#####################################################")

    '''
    save_model(encoder, "mymodel_encoder.pth")
    save_model(decoder1, "mymodel_decoder1.pth")
    save_model(decoder2, "mymodel_decoder2.pth")
    '''

    return train_loss, val_loss


# TEST
def test_model3():
    batch_size = 1
    epochs = 100
    e = 1
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    encoder = Encoder().to(device)
    decoder_starting_layer = Decoder().to(device)
    decoder = Decoder(final_layer=True).to(device)

    PATH1 = "mymodel_encoder.pth"
    PATH2 = "mymodel_decoder1.pth"
    PATH3 = "mymodel_decoder2.pth"

    encoder.load_state_dict(torch.load(PATH1))
    decoder_starting_layer.load_state_dict(torch.load(PATH2))
    decoder.load_state_dict(torch.load(PATH3))

    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    te_loader = GraphDataset()
    test_loader = DataLoader(te_loader, batch_size=batch_size)

    train_loss = []
    val_loss = []

    optimality_val = []
    optimality_train = []

    for epoch in range(epochs):

        if epoch % e == 0:
            print(f"Epoch {epoch}...")

        # TESTING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(test_loader):

            encoder.eval()
            decoder_starting_layer.eval()
            decoder.eval()

            linegraph.x = linegraph.features.view(linegraph.num_nodes, -1).type(torch.float)

            logprobs = 0
            reward = 0

            # this set is for nodes in linegraph
            node_set = set()

            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                encoded_features = encoder(linegraph.x, linegraph.edge_index)

                # create an empty graph
                G = nx.Graph()

                starting_node = decoder_starting_layer(encoded_features)
                starting_node = torch.argmax(starting_node, dim=0)
                previous_node = starting_node

                # create adj matrix from edge index
                adj = torch.zeros(encoded_features.shape[0], encoded_features.shape[0])
                for i in range(len(linegraph.edge_index[0])):
                    u = linegraph.edge_index[0][i]
                    v = linegraph.edge_index[1][i]
                    adj[u][v] = 1
                    adj[v][u] = 1

                node_probs_entropy = torch.zeros(encoded_features.shape[0])

                # mst_wt = torch.tensor(mst_wt,requires_grad=True)

                for i in range(n_nodes - 1):

                    probs = decoder(encoded_features, starting_node, previous_node, node_set, adj)

                    # sample next node from probs
                    # sampler = Categorical(probs)
                    # next_node = sampler.sample()
                    # print(probs)
                    next_node = torch.argmax(probs, dim=0)

                    node_probs_entropy[next_node] = probs[next_node]

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node

                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    for node in range(probs.shape[0]):
                        reward += (probs[node] * linegraph.x[node])

                    # add the current node in the node set
                    node_set.add(int(next_node))

            # loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            loss = C31 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C32 * criterion2(reward, mst_wt)

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))

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
                print("")
            # '''

        '''
        if epoch % e == 0:
            save_model(encoder)
        '''
    # print(f"optimality_val = {optimality_val}")

    optimality_avg = torch.tensor(optimality_val).float()

    print("#####################################################")
    print("Final Report")
    print(f"Total Number of Different Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print("#####################################################")

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
'''
Using Gumbel Softmax
'''


C1 = 1.25
C2 = 1.


def train(n_nodes, n_edges, use_adj):
    batch_size = 1

    epochs = 1000

    e1 = 100  # after every e1 epochs print graphs
    e2 = 20  # after every e2 epoch save model to same file

    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'

    maxi = 1000
    saved_epoch = -1
    
    encoder = Encoder().to(device)
    decoder1 = Decoder().to(device)
    decoder2 = Decoder(final_layer=True).to(device)

    tr_loader = GraphDataset(n_nodes, n_edges)
    train_loader = DataLoader(tr_loader, batch_size=batch_size)
    
    optimizer_encoder = torch.optim.SGD(encoder.parameters(), lr=1e-2, weight_decay=0.01)
    optimizer_decoder1 = torch.optim.SGD(decoder1.parameters(), lr=1e-2, weight_decay=0.01)
    optimizer_decoder2 = torch.optim.SGD(decoder2.parameters(), lr=1e-2, weight_decay=0.01)
    
    scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_encoder, step_size=200, gamma=0.1)
    scheduler_decoder1 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder1, step_size=200, gamma=0.1)
    scheduler_decoder2 = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_decoder2, step_size=200, gamma=0.1)
    
    train_loss = []
    val_loss = []
    
    optimality_val = []
    optimality_val_complete = [] # of all the generated mst's [doesn't matter whether proper or not]
    optimality_train = []
    
    criterion1 = nn.BCELoss()
    criterion2 = nn.L1Loss()

    for epoch in range(epochs):
        
        if epoch % e1 == 0:
            print(f"Epoch {epoch}...")
        
        early_stop = False
        
        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):
            
            encoder.train()
            decoder1.train()
            decoder2.train()
            
            optimizer_encoder.zero_grad()# Clear gradients.
            optimizer_decoder1.zero_grad()# Clear gradients.
            optimizer_decoder2.zero_grad()# Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
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
            
            logprobs = 0
            reward = 0
            
            #this set is for nodes in linegraph
            node_set = set()
            
            starting_node = None
            previous_node = None
            
            #mst_wt = torch.tensor(mst_wt, requires_grad=True)
            
            node_probs_entropy = torch.zeros(encoded_features.shape[0])
            
            for i in range(n_nodes-1):
                
                if(i == 0):
                    x = encoded_features.mean(dim=0).squeeze()
                else:
                    x = out
                
                out = decoder1(encoded_features, node_set, starting_node, previous_node, x, adj, use_adj)
                logits = decoder2(encoded_features, node_set, starting_node, previous_node, out, adj, use_adj)
                
                # sample next node from probs
                #sampler = Categorical(probs)
                #next_node = sampler.sample()
                #next_node = torch.argmax(probs, dim=0)
                #next_node = (F.gumbel_softmax(logits, tau=1, hard=True)@torch.arange(encoded_features.shape[0], dtype=torch.float32)).int()

                '''
                probs = F.gumbel_softmax(logits, tau=1, hard=False)
                
                next_node = torch.argmax(probs)
                
                node_probs_entropy[next_node] = probs[next_node]
                '''
                h = F.gumbel_softmax(logits, tau=1, hard=True)
                node_probs_entropy += h
                next_node = torch.argmax(h)


                # once we have next node we make it our starting node and begin the algorithm all over again
                previous_node = next_node
                
                if(starting_node == None):
                    starting_node = next_node
                
                # add edge between nodes in primal to constructed tree
                u = int(line_graph_nodes[next_node][0])
                v = int(line_graph_nodes[next_node][1])
                G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])
                
                reward += (linegraph.x[next_node])
                
                #add the current node in the node set
                node_set.add(int(next_node))
                
            #loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            #loss = C1 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C2 * criterion2(reward, mst_wt)
            loss = nn.L1Loss()(node_probs_entropy, target_entropy.squeeze()) * nn.L1Loss()(reward, mst_wt)
            #optimality_train.append(calc_optimality(mst,G,n_nodes, mst_wt))
            
            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            optimizer_decoder1.step()  # Update parameters based on gradients.
            optimizer_decoder2.step()  # Update parameters based on gradients.
            
            scheduler_encoder.step()
            scheduler_decoder1.step()
            scheduler_decoder2.step()
            
            train_loss.append(loss.item())
            
            #early_stop = check_early_stop(train_loss, epoch)
            
            
        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt, target_entropy) in enumerate(train_loader):
            
            encoder.eval()
            decoder1.eval()
            decoder2.eval()
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            logprobs = 0
            reward = 0
            
            #this set is for nodes in linegraph
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
                
                #mst_wt = torch.tensor(mst_wt,requires_grad=True)
                
                node_probs_entropy = torch.zeros(encoded_features.shape[0])
                
                for i in range(n_nodes-1):
                    
                    if(i == 0):
                        x = encoded_features.mean(dim=0).squeeze()
                    else:
                        x = out
                    
                    out = decoder1(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    logits = decoder2(encoded_features, node_set, starting_node, previous_node, out, use_adj)
                    
                    # sample next node from probs
                    #sampler = Categorical(probs)
                    #next_node = sampler.sample()
                    #print(probs)
                    #next_node = (F.gumbel_softmax(logits, tau=1, hard=True)@torch.arange(encoded_features.shape[0], dtype=torch.float32)).int()

                    '''
                    probs = F.gumbel_softmax(logits, tau=1, hard=False)
                
                    next_node = torch.argmax(probs)
                
                    node_probs_entropy[next_node] = probs[next_node]
                    '''
                    h = F.gumbel_softmax(logits, tau=1, hard=True)
                    node_probs_entropy += h
                    next_node = torch.argmax(h)

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node
                    
                    if(starting_node == None):
                        starting_node = next_node
                    
                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    reward += (linegraph.x[next_node])
                    
                    #add the current node in the node set
                    node_set.add(int(next_node))
                
            #loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            #loss = C1 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C2 * criterion2(reward, mst_wt)
            loss = nn.L1Loss()(node_probs_entropy, target_entropy.squeeze()) * nn.L1Loss()(reward, mst_wt)

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))
            optimality_val_complete.append(calc_optimality(mst, G, n_nodes, mst_wt, calc_all=True))
            
            val_loss.append(loss.item())
            #'''
            if epoch % e1 == 0:
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
            #'''

        '''        
        if early_stop:
            print("\nStopping Ealry !!\n")
            break
        '''
        if epoch % e2 == 0:
            if epoch == 0:
                print(f"Model saved at epoch {epoch}")
                save_model(encoder, "mymodel_encoder.pth")
                save_model(decoder1, "mymodel_decoder1.pth")
                save_model(decoder2, "mymodel_decoder2.pth")
            else:
                if val_loss[epoch] <= maxi:
                    maxi = val_loss[epoch]
                    saved_epoch = epoch
                    print(f"Model saved at epoch {epoch}")
                    save_model(encoder, "mymodel_encoder.pth")
                    save_model(decoder1, "mymodel_decoder1.pth")
                    save_model(decoder2, "mymodel_decoder2.pth")


    #print(f"optimality_val = {optimality_val}")
    
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
    
    return train_loss, val_loss


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
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            logprobs = 0
            reward = 0
            
            #this set is for nodes in linegraph
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
                
                #mst_wt = torch.tensor(mst_wt,requires_grad=True)
                
                for i in range(n_nodes-1):
                    
                    if(i == 0):
                        x = encoded_features.mean(dim=0).squeeze()
                    else:
                        x = out
                    
                    out = decoder1(encoded_features, node_set, starting_node, previous_node, x, use_adj)
                    logits = decoder2(encoded_features, node_set, starting_node, previous_node, out, use_adj)
                    
                    # sample next node from probs
                    #sampler = Categorical(probs)
                    #next_node = sampler.sample()
                    #print(probs)
                    #next_node = (F.gumbel_softmax(logits, tau=1, hard=True)@torch.arange(encoded_features.shape[0], dtype=torch.float32)).int()

                    '''
                    probs = F.gumbel_softmax(logits, tau=1, hard=False)
                
                    next_node = torch.argmax(probs)
                
                    node_probs_entropy[next_node] = probs[next_node]
                    '''

                    h = F.gumbel_softmax(logits, tau=1, hard=True)
                    node_probs_entropy += h
                    next_node = torch.argmax(h)

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node
                    
                    if(starting_node == None):
                        starting_node = next_node
                    
                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])

                    reward += (linegraph.x[next_node])
                    
                    #add the current node in the node set
                    node_set.add(int(next_node))
                
            #loss, C = loss_func(reward, logprobs, G, n_nodes, PENALTY, mst_wt, -1 if epoch==0 else C)
            #loss = C1 * criterion1(node_probs_entropy, target_entropy.squeeze()) + C2 * criterion2(reward, mst_wt)
            loss = nn.L1Loss()(node_probs_entropy, target_entropy.squeeze()) * nn.L1Loss()(reward, mst_wt)

            optimality_val.append(calc_optimality(mst, G, n_nodes, mst_wt))
            optimality_val_complete.append(calc_optimality(mst, G, n_nodes, mst_wt, calc_all=True))
            
            val_loss.append(loss.item())
            #'''
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
            #'''
                
        
        '''
        if epoch % e == 0:
            save_model(encoder)
        '''
    #print(f"optimality_val = {optimality_val}")
    
    optimality_avg = torch.tensor(optimality_val).float()
    optimality_avg_complete = torch.tensor(optimality_val_complete).float()
    
    print("\n#####################################################")
    print("Final Report")
    print(f"Total Number of Randomly Generated Graphs = {epochs}")
    print(f"Total Number of MST's = {len(torch.where(optimality_avg > 0)[0])}")
    print(f"Avg Optimality of generated MST's = {optimality_avg[torch.where(optimality_avg > 0)[0]].mean(dim=0)}")
    print(f"Avg Optimality of all generated MST's (proper/improper) = {optimality_avg_complete.mean(dim=0)}")
    print("#####################################################")
