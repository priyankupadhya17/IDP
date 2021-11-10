from generate_graph import create_graph_dataset
import torch
from model_copy import Encoder, Decoder
from dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from generate_graph import show_graph
from networkx.algorithms.tree.recognition import is_tree
from torch import nn
from networkx.classes.function import number_of_nodes
from networkx.linalg.graphmatrix import adjacency_matrix
from networkx.algorithms.similarity import graph_edit_distance
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


def calc_optimality(mst, G, n_nodes, mst_wt):
    
    #by default optimality = -1
    optimality = -1
    
    sum_of_weights = 0
    for (_, _, w) in G.edges(data=True):
        sum_of_weights += w['weight']
        
    print(f"constructed mst total_wt = {sum_of_weights}")
    print(f"actual mst total_wt= {mst_wt}")
    
    if is_tree(G) and G.number_of_nodes() == n_nodes:
        optimality = sum_of_weights / mst_wt
        print("*********************************")
        print(f"optimality = {optimality}")
        print("*********************************")
    
    return optimality


def loss_func(reward, prob, graph, n_nodes):
    
    TREE_PENALTY = 10
    N_NODES_PENALTY = 5
    
    loss = -torch.mean(reward * prob)
    
    #print(f"is_tree = {is_tree(graph)}")
    #print(f"graph.number_of_nodes() = {graph.number_of_nodes()}, n_nodes = {n_nodes}")
    
    #loss = loss * (1 if is_tree(graph) else TREE_PENALTY)
    #loss = loss * (1 if graph.number_of_nodes() == n_nodes else N_NODES_PENALTY)
    
    loss = loss + (0 if is_tree(graph) else 1000)
    
    return loss
    


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

def save_model(model):
    current_directory = os.getcwd()
    filename = os.path.join(current_directory, "mymodel.pth")
    torch.save(model.state_dict(), filename)
    
def train():
    batch_size = 1
    epochs = 1000
    e = 100
    eps = 1e-15
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.001)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.001)
    
    train_loss = []
    val_loss = []
    
    optimality_val = []
    optimality_train = []

    for epoch in range(epochs):
        
        if epoch % e == 0:
            print(f"Epoch {epoch}...")
        
        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt) in enumerate(train_loader):
            
            encoder.train()
            decoder.train()
            
            optimizer_encoder.zero_grad()  # Clear gradients.
            optimizer_decoder.zero_grad()  # Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            # total number of nodes in primal graph
            n_nodes = graph.num_nodes
            
            # encoder returns probs of starting node
            encoded_features = encoder(linegraph.x, linegraph.edge_index)
            
            # create an empty graph
            G = nx.Graph()
            
            logprobs = 0
            reward = 0
            
            node_set = set()
            
            starting_node = encoded_features.argmax(dim=0).squeeze()
            u = int(line_graph_nodes[starting_node][0])
            v = int(line_graph_nodes[starting_node][1])
            G = add_nodes_in_graph(G, u, v, linegraph.x[starting_node])
            node_set.add(int(starting_node))
            
            previous_node = None
            
            
            for i in range(n_nodes-2):
                
                out, next_node = decoder(encoded_features, node_set, starting_node, previous_node)
                
                # once we have next node we make it our starting node and begin the algorithm all over again
                previous_node = next_node
                
                node_set.add(int(next_node))
                
                # add edge between nodes in primal to constructed tree
                u = int(line_graph_nodes[next_node][0])
                v = int(line_graph_nodes[next_node][1])
                G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])
                
                #reward += calc_reward(G, n_nodes)
                reward += G[u][v]['weight'] #* (100 if((u in node_set) and (v in node_set)) else 1)
                logprobs += torch.log(out + eps).mean()
                
                
            loss = loss_func(reward, logprobs, G, n_nodes)
            
            optimality_train.append(calc_optimality(mst,G,n_nodes, mst_wt))
            
            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            optimizer_decoder.step()  # Update parameters based on gradients.
            
            train_loss.append(loss.item())
            
            
            
        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes, mst_wt) in enumerate(train_loader):
            
            encoder.eval()
            decoder.eval()
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            logprobs = 0
            reward = 0
            
            node_set = set()
            
            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                encoded_features = encoder(linegraph.x, linegraph.edge_index)

                # create an empty graph
                G = nx.Graph()
                
                starting_node = encoded_features.argmax(dim=0).squeeze()
                u = int(line_graph_nodes[starting_node][0])
                v = int(line_graph_nodes[starting_node][1])
                G = add_nodes_in_graph(G, u, v, linegraph.x[starting_node])
                node_set.add(int(starting_node))
                
                previous_node = None
            
                
                for i in range(n_nodes-2):
                
                    out, next_node = decoder(encoded_features, node_set, starting_node, previous_node)

                    # once we have next node we make it our starting node and begin the algorithm all over again
                    previous_node = next_node
                    
                    node_set.add(int(next_node))
                    
                    # add edge between nodes in primal to constructed tree
                    u = int(line_graph_nodes[next_node][0])
                    v = int(line_graph_nodes[next_node][1])
                    G = add_nodes_in_graph(G, u, v, linegraph.x[next_node])
                    
                    #reward += calc_reward(G, n_nodes)
                    reward += G[u][v]['weight'] #* (100 if((u in node_set) and (v in node_set)) else 1)
                    logprobs += torch.log(out + eps).mean()
                    
                
            loss = loss_func(reward, logprobs, G, n_nodes)
            
            optimality_val.append(calc_optimality(mst,G,n_nodes, mst_wt))
            
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
            #'''
                
        
        '''
        if epoch % e == 0:
            save_model(encoder)
        '''
    print(f"optimality_val = {optimality_val}")
    
    return train_loss, val_loss

