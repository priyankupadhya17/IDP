from generate_graph import create_graph_dataset
import torch
from model import Model
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


def loss_func(graph, mst, requires_grad):
    
    mst = to_networkx(mst, to_undirected=True)
    #show_graph(mst)
    
    n_nodes_graph = number_of_nodes(graph)
    n_nodes_mst = number_of_nodes(mst)
    
    l1 = nn.L1Loss()
    
    MAX_INDICATOR_VALUE1 = torch.tensor(1000, dtype=float, requires_grad=requires_grad)
    MIN_INDICATOR_VALUE1 = torch.tensor(0, dtype=float, requires_grad=requires_grad)
    
    
    MAX_INDICATOR_VALUE2 = torch.tensor(1000, dtype=float, requires_grad=requires_grad)
    MIN_INDICATOR_VALUE2 = torch.tensor(0, dtype=float, requires_grad=requires_grad)
    
    #indicator_function = 0 if is_tree(graph) else 1000
    indicator_function1 = l1(MAX_INDICATOR_VALUE1, MAX_INDICATOR_VALUE1 if is_tree(graph) else MIN_INDICATOR_VALUE1)
    indicator_function2 = l1(MAX_INDICATOR_VALUE2, MAX_INDICATOR_VALUE2 if n_nodes_graph==n_nodes_mst else MIN_INDICATOR_VALUE2)    
    
    sum_of_weights = 0
    for (_, _, w) in graph.edges(data=True):
        sum_of_weights += w['weight']
    
    
    total_loss = indicator_function1 + indicator_function2 + \
    l1(torch.tensor(sum_of_weights, dtype=float, requires_grad=requires_grad), torch.tensor(0, dtype=float, requires_grad=requires_grad))
    return total_loss
    

def save_model(model):
    current_directory = os.getcwd()
    filename = os.path.join(current_directory, "mymodel.pth")
    torch.save(model.state_dict(), filename)
    
def train():
    batch_size = 1
    epochs = 1000
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = Model(hidden_dim=3, feature_dim=1).to(device)

    print(model)

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        
        # TRAINING
        for batch, (graph, linegraph, mst, line_graph_nodes) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            out = model(linegraph.x, linegraph.edge_index)  # Perform a single forward pass.
            
            # this will return a graph/tree that is created using output nodes(matched to line_graph nodes which are 
            # further matched to graph)
            G = create_graph_from_model_output(out, line_graph_nodes, graph)
            
            #show_graph(G)
            
            loss = loss_func(G, mst, True)
            train_loss.append(loss.item())
            
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            
        # VALIDATION
        for batch, (graph, linegraph, mst, line_graph_nodes) in enumerate(train_loader):
            model.eval()
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            
            with torch.no_grad():
                # Perform a single forward pass.
                out = model(linegraph.x, linegraph.edge_index)  
                # this will return a graph/tree that is created using output nodes(matched to line_graph nodes which are 
                # further matched to graph)
                G = create_graph_from_model_output(out, line_graph_nodes, graph)
                loss = loss_func(G, mst, False)
                val_loss.append(loss.item())
            
            if epoch % 100 == 0:
                print("Original Graph:")
                show_graph(to_networkx(graph, to_undirected=True))
                print("LineGraph:")
                show_graph(to_networkx(linegraph, to_undirected=True))
                print("Output MST:")
                show_graph(G)
                print("Actual MST:")
                show_graph(to_networkx(mst, to_undirected=True))
            
        if epoch % 100 == 0:
            save_model(model)
    
    return train_loss, val_loss


if __name__ == '__main__':
    #create_graph_dataset(1)
    #train()
    print(torch.__version__)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(device)

