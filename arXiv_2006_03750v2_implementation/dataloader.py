import os
import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import to_networkx, from_networkx
from arXiv_2006_03750v2_implementation.generate_graph import create_graph_dataset


class GraphDataset(Dataset):
    def __init__(self, root=None, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    def len(self):
        return 1

    def get(self, idx):
        graph, linegraph, mst = create_graph_dataset()
        return from_networkx(graph), from_networkx(linegraph), from_networkx(mst)
