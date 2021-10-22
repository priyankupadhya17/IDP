from torch.nn import Linear, ReLU, Module
from torch_geometric.nn import Sequential, GCNConv, TopKPooling


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = GCNConv(in_channels=1, out_channels=3)
        self.conv2 = GCNConv(in_channels=3, out_channels=3)
        self.conv3 = GCNConv(in_channels=3, out_channels=3)
        self.linear = Linear(in_features=3, out_features=1)
        self.relu = ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))

        return x

