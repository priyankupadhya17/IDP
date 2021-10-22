from arXiv_2006_03750v2_implementation.generate_graph import create_graph_dataset
import torch
#from arXiv_2006_03750v2_implementation.model import Model
#from arXiv_2006_03750v2_implementation.dataloader import GraphDataset
#from torch.utils.data import DataLoader


def train():
    batch_size = 1
    #device = 0 if torch.cuda.is_available() else 'cpu'
    #model = Model().to(device)

    #print(model)

    tr_loader = GraphDataset()
    train_loader = DataLoader(tr_loader, batch_size=batch_size)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epochs in range(1):
        for batch, (graph, linegraph, mst) in enumerate(train_loader):
            #model.train()
            #optimizer.zero_grad()  # Clear gradients.

            linegraph_edge_index = linegraph.edge_index
            linegraph_features = linegraph.x

            print(linegraph_features)
            print(linegraph_edge_index)

            '''
            out = model(x, edge_index_tensor)  # Perform a single forward pass.
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

