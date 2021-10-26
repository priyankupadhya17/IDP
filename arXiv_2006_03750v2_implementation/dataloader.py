import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx, from_networkx
from generate_graph import create_graph_dataset


class GraphDataset(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    def len(self):
        return 1

    def get(self, idx):
        graph, linegraph, mst, line_graph_nodes = create_graph_dataset()
        
        graph_from_networkx = from_networkx(graph)
        linegraph_from_networkx = from_networkx(linegraph) 
        mst_from_networkx = from_networkx(mst)
        
        
        return graph_from_networkx, linegraph_from_networkx, mst_from_networkx, line_graph_nodes
