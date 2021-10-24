from generate_graph import create_graph_dataset
import torch
from model import Model
from dataloader import GraphDataset
from torch_geometric.loader import DataLoader


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
        for batch, (graph, linegraph, mst) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()  # Clear gradients.
            
            linegraph.x = linegraph.features.view(linegraph.num_nodes,-1).type(torch.float)
            
            print(graph)
            print(linegraph)
            print(mst)
            
            out = model(linegraph.x, linegraph.edge_index)  # Perform a single forward pass.
            '''
            loss = criterion(out[train_mask], y[train_mask])  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            '''




if __name__ == '__main__':
    #create_graph_dataset(1)
    #train()
    print(torch.__version__)
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(device)

