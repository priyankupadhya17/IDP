from generate_graph import create_graph_dataset
import torch
from model import Encoder, Decoder
from dataloader import GraphDataset
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx
from generate_graph import show_graph
from networkx.algorithms.tree.recognition import is_tree
from torch import nn
from networkx.classes.function import number_of_nodes
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


def loss_func(reward, prob):
    return torch.sum(reward*prob)
    


def calc_reward(graph, requires_grad, n_nodes):
    
    MAX_INDICATOR_VALUE = 1000
    MIN_INDICATOR_VALUE = 0
    
    # indicator_function = 0 if is_tree(graph) else 1000
    indicator_function = MIN_INDICATOR_VALUE if is_tree(graph) else MAX_INDICATOR_VALUE
    
    sum_of_weights = 0
    for (_, _, w) in graph.edges(data=True):
        sum_of_weights += w['weight']
    
    
    reward = -torch.tensor(indicator_function + sum_of_weights, dtype=float ,requires_grad=requires_grad)
    return reward
    

def add_nodes_in_graph(graph, u, v, wt):
    graph.add_edges_from([(int(u), int(v),{'weight': int(wt)})])
    return graph

def save_model(model):
    current_directory = os.getcwd()
    filename = os.path.join(current_directory, "mymodel.pth")
    torch.save(model.state_dict(), filename)
    
def train():
    batch_size = 1
    epochs = 10000
    e = 200
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    encoder = Encoder().to(device)
    decoder = Decoder(hidden_dim=3, feature_dim=1)
    

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr=0.001)
    
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        
        if epoch % e == 0:
            print(f"Epoch {epoch}...")
        
        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes) in enumerate(train_loader):
            
            encoder.train()
            
            optimizer_encoder.zero_grad()  # Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            # total number of nodes in primal graph
            n_nodes = graph.num_nodes
            
            # encoder returns probs of starting node
            out = encoder(linegraph.x, linegraph.edge_index)
            
            # create an empty graph
            G = nx.Graph()
            
            # select a starting node 
            starting_node = torch.argmax(out, dim=0)
            
            reward = []
            probs = []
            
            node_set = set()
            
            for i in range(n_nodes-1):
                
                # add edge between nodes in primal to constructed tree
                u = line_graph_nodes[starting_node][0]
                v = line_graph_nodes[starting_node][1]
                G = add_nodes_in_graph(G, u, v, graph.weight[starting_node])
                
                node_set.add(int(u))
                node_set.add(int(v))
                
                #add prob to probs
                probs.append(out[starting_node])
                
                if len(node_set) != n_nodes:
                    # find the next node(of linegraph) from the decoder
                    next_node = decoder.decode(starting_node, out, linegraph.edge_index, node_set, line_graph_nodes)
                    starting_node = next_node
                
                reward.append(calc_reward(G, requires_grad=True, n_nodes=n_nodes))
                
            loss = loss_func(torch.tensor(reward, dtype=float, requires_grad=True), torch.tensor(probs, dtype=float, requires_grad=True))
            loss.backward()  # Derive gradients.
            optimizer_encoder.step()  # Update parameters based on gradients.
            train_loss.append(loss.item())
            
            
        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes) in enumerate(train_loader):
            
            encoder.eval()
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            reward = []
            probs = []
            
            node_set = set()
            
            with torch.no_grad():
                # total number of nodes in primal graph
                n_nodes = graph.num_nodes

                # encoder returns probs of starting node
                out = encoder(linegraph.x, linegraph.edge_index)

                # create an empty graph
                G = nx.Graph()

                # select a starting node 
                starting_node = torch.argmax(out, dim=0)
                
                
                for i in range(n_nodes-1):

                    # add edge between nodes in primal to constructed tree
                    u = line_graph_nodes[starting_node][0]
                    v = line_graph_nodes[starting_node][1]
                    G = add_nodes_in_graph(G, u, v, graph.weight[starting_node])
                    
                    node_set.add(int(u))
                    node_set.add(int(v))
                    
                    #add prob to probs
                    probs.append(out[starting_node])
                    
                    if len(node_set) != n_nodes:
                        # find the next node(of linegraph) from the decoder
                        next_node = decoder.decode(starting_node, out, linegraph.edge_index, node_set, line_graph_nodes)
                        starting_node = next_node

                    reward.append(calc_reward(G, requires_grad=True, n_nodes=n_nodes))
            
            loss = loss_func(torch.tensor(reward), torch.tensor(probs))
            val_loss.append(loss.item())
            
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
                
        
        '''
        if epoch % e == 0:
            save_model(encoder)
        '''
    
    return train_loss, val_loss

