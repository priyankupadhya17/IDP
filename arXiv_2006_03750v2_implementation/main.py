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


def loss_func(graph, mst):
    l1 = nn.L1Loss()
    
    MAX_INDICATOR_VALUE = torch.tensor(1000, dtype=float, requires_grad=True)
    MIN_INDICATOR_VALUE = torch.tensor(0, dtype=float, requires_grad=True)
    
    #indicator_function = 0 if is_tree(graph) else 1000
    indicator_function = l1(MAX_INDICATOR_VALUE, MAX_INDICATOR_VALUE if is_tree(graph) else MIN_INDICATOR_VALUE)
    
    sum_of_weights = 0
    for (_, _, w) in graph.edges(data=True):
        sum_of_weights += w['weight']
    
    mst = to_networkx(mst, to_undirected=True)
    show_graph(mst)
    
    total_loss = indicator_function + l1(torch.tensor(sum_of_weights, dtype=float, requires_grad=True), torch.tensor(0, dtype=float, requires_grad=True))
    return total_loss
    

def train():
    batch_size = 1
    device = 0 if torch.cuda.is_available() else 'cpu'
    model = Model(hidden_dim=3, feature_dim=1).to(device)

    print(model)

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epochs in range(1):
        for batch, (graph, linegraph, mst, line_graph_nodes) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            
            out = model(linegraph.x, linegraph.edge_index)  # Perform a single forward pass.
            
            # this will return a graph/tree that is created using output nodes(matched to line_graph nodes which are 
            # further matched to graph)
            G = create_graph_from_model_output(out, line_graph_nodes, graph)
            
            show_graph(G)
            
            loss = loss_func(G, mst)
            print(loss.item())
            
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.


if __name__ == '__main__':
    #create_graph_dataset(1)
    #train()
    print(torch.__version__)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(device)

