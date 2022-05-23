import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from torch.nn import Linear, ReLU, Module, Sequential, Softmax, Parameter, LeakyReLU, Sigmoid
from torch_geometric.nn import Sequential, GAT, GATConv, GCNConv, SAGEConv, GraphConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import random
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import math


path_loc = "/Users/priyankupadhya/Desktop/mimic_dataset/train_copy"
#path_loc = "/Users/priyankupadhya/Desktop/mimic_dataset/train_copy2"
save_path = "/Users/priyankupadhya/Desktop/mimic_dataset/saves"
test_path = "/Users/priyankupadhya/Desktop/mimic_dataset/test_copy"
#test_path = "/Users/priyankupadhya/Desktop/mimic_dataset/test_copy2"

TRAIN_VAL_DATASET_SIZE = 17455


def read_csv(loc, normalize=False, returntype='numpy'):
    '''
    returntype = 'numpy' or 'df'(if returntype is 'df' do not forget to convert the df to numpy later on)
    '''

    if returntype != 'numpy' and returntype != 'df':
        raise ValueError('returntype is not VALID ; possible values = [numpy, df]')

    df = pd.read_csv(loc)
    # print(df.isnull().values.any())
    if normalize:

        for column in df:
            if df[column].std() < 0.001:
                df[column].values[:] = 0
            else:
                df[column] = (df[column] - df[column].mean()) / (df[column].std())

    # print(df.isnull().values.any())

    if returntype == 'numpy':
        return df.to_numpy()
    else:
        return df


def read_mimic_dataset(loc, returntype='numpy'):
    # patient is a dict that maps patient to linear_imputed_vals and treatment array
    patient = {}

    cnt = 0

    for root, dirs, files in os.walk(loc):

        if "patient" in root:

            patient_id = root.split('/')[len(root.split('/')) - 1]

            d = {}
            for file in files:
                if file == "ts_vals_linear_imputed.csv":
                    d.update({"ts_vals_linear_imputed.csv": read_csv(os.path.join(root, file), normalize=True, returntype=returntype)})
                elif file == "ts_treatment.csv":
                    d.update({"ts_treatment.csv": read_csv(os.path.join(root, file), returntype=returntype)})
                elif file == "statics.csv":
                    d.update({"statics.csv": read_csv(os.path.join(root, file), returntype=returntype)})
                elif file == "rolling_tasks_binary_multilabel.csv":
                    d.update({"rolling_tasks_binary_multilabel.csv": read_csv(os.path.join(root, file), returntype=returntype)})

            d.update({"patient_id": patient_id})

            patient[cnt] = d

            cnt = cnt + 1
            if cnt % 100 == 0:
                print(f"{cnt} files completed")

    return patient


def compareMortality(patient):
    total_0 = 0
    total_1 = 0
    new_patient_indices_mortality_only_1 = []
    new_patient_indices_mortality_only_0 = []

    for p_index, p in patient.items():
        labels = p["rolling_tasks_binary_multilabel.csv"][:, 0]
        mortality_ones = len(np.where(labels > 0.0)[0])
        mortality_zeros = len(np.where(labels < 1.0)[0])
        total_0 += mortality_zeros
        total_1 += mortality_ones

        # create new dataset of patients and include patients with mortality 1
        if mortality_ones > 0:
            new_patient_indices_mortality_only_1.append(p_index)
        else:
            new_patient_indices_mortality_only_0.append(p_index)


        print(f"{p['patient_id']}: Mortality 0's:{mortality_zeros} , 1's:{mortality_ones}")

    print(f"For all patients: Count of 0's = {total_0} and Count of 1's = {total_1}")

    return new_patient_indices_mortality_only_0, new_patient_indices_mortality_only_1


class MimicDataset(Dataset):
    def __init__(self, patient, n=30, k_forward=2, k_backward=2, k_self=1):
        """
        patient = dict that describes all the relevant csv files and their info for all patients in directory
        n = no. of nodes in the graph
        k_forward = no. of nodes connected in forward direction [eg:node2-->node3 and node2-->node4 for k_forward=2]
        k_backward = no. of nodes connected in backward direction [eg:node2-->node1 and node2-->node1 for k_backward=2]
        k_self = 1 since node is connected to itself
        """

        self.patient = patient
        self.n = n
        self.k_backward = k_backward
        self.k_forward = k_forward
        self.k_self = k_self
        self.files = ["ts_vals_linear_imputed.csv",
                      "ts_treatment.csv",
                      "statics.csv",
                      "rolling_tasks_binary_multilabel.csv"
                      ]

    def __len__(self):
        return len(self.patient)

    def __getitem__(self, idx):

        """
        should return patient_id, features, edge_index and targets for a single patient
        """

        p_data = self.patient[idx]

        patient_id = p_data["patient_id"]

        total_timestamps = len(p_data[self.files[0]])

        d = total_timestamps / self.n

        features = []
        targets = []
        cnt = 0
        j = 0
        while j < total_timestamps:
            cnt += 1
            k = int(j)
            f1 = p_data[self.files[0]][k].squeeze()
            f2 = p_data[self.files[1]][k].squeeze().astype('float')

            f3 = p_data[self.files[2]].squeeze()
            f3[0] = f3[0] / 100
            f3[2] = f3[2] / 5
            f3[3] = f3[3] / 5
            f3[4] = f3[4] / 3
            f3[5] = f3[5] / 5

            targets.append(p_data[self.files[3]][k][0].squeeze())

            features.append(np.concatenate((f1, f2, f3)))

            j = j + d
            if cnt == self.n:
                break

        # make edge list
        edge_index = []
        e1 = []
        e2 = []
        for i in range(len(targets)):
            for j in range(1, self.k_backward + 1):
                if i - j >= 0:
                    e1.append(i)
                    e2.append(i - j)
            for j in range(1, self.k_forward + 1):
                if i + j < len(targets):
                    e1.append(i)
                    e2.append(i + j)
        edge_index.append(e1)
        edge_index.append(e2)

        data = {"patient_id": patient_id,
                "features": torch.from_numpy(np.array(features, dtype=float)),
                "edge_index": torch.from_numpy(np.array(edge_index, dtype=np.compat.long)),
                "targets": torch.from_numpy(np.array(targets, dtype=float))}

        return data


class MimicModel(Module):
    def __init__(self, num_edges, degree_per_node, num_features):
        super(MimicModel, self).__init__()

        # self.edge_weight = Parameter(torch.normal(mean=0.0, std=1.0/degree_per_node, size=(num_edges,)))
        self.edge_weight = Parameter(torch.ones(size=(num_edges,)))

        self.conv1 = GCNConv(num_features, 50)
        self.conv2 = GCNConv(50, 50)
        self.conv3 = GCNConv(50, 30)
        self.conv4 = GCNConv(30, 30)
        self.conv5 = GCNConv(30, 10)
        self.conv6 = GCNConv(10, 10)
        self.conv7 = GCNConv(10, 1)

        self.relu = ReLU(inplace=True)
        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index, self.edge_weight)
        x = self.relu(x)

        x = self.conv2(x, edge_index, self.edge_weight)
        x = self.relu(x)
        x = F.dropout(x)

        x = self.conv3(x, edge_index, self.edge_weight)
        x = self.relu(x)

        x = self.conv4(x, edge_index, self.edge_weight)
        x = self.relu(x)
        x = F.dropout(x)

        x = self.conv5(x, edge_index, self.edge_weight)
        x = self.relu(x)

        x = self.conv6(x, edge_index, self.edge_weight)
        x = self.relu(x)
        x = F.dropout(x)

        x = self.conv7(x, edge_index, self.edge_weight)

        return x


def save_model(model, name):
    filename = os.path.join(save_path, name)
    torch.save(model.state_dict(), filename)


def train(patient_dict, k_back, k_for, k_self=1, n=30):
    hparams = {'batch_size': 1, 'lr': 0.0001, 'epochs': 100, 'k_backward': k_back, 'k_forward': k_for, 'k_self': k_self,
               'n': n, 'train_val_split': 0.80, 'wt_decay': 0.01}

    maxi_loss = 99999999

    dataset_size = len(patient_dict)
    indices = list(range(dataset_size))
    split = int(dataset_size * hparams['train_val_split'])
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    mimicdataset = MimicDataset(patient_dict,
                                n=hparams['n'],
                                k_forward=hparams['k_forward'],
                                k_backward=hparams['k_backward'],
                                k_self=hparams['k_self'])

    num_edges = mimicdataset[0]['edge_index'].squeeze().shape[1]
    degree_per_node = hparams['k_backward'] + hparams['k_forward'] + hparams['k_self']
    num_features = mimicdataset[0]['features'].squeeze().shape[1]

    train_dataloader = DataLoader(mimicdataset, batch_size=hparams['batch_size'], sampler=train_sampler)
    val_dataloader = DataLoader(mimicdataset, batch_size=hparams['batch_size'], sampler=val_sampler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MimicModel(num_edges, degree_per_node, num_features).to(device).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['wt_decay'])

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loss = []
    val_loss = []
    val_acc = []
    val_prec = []
    val_rec = []

    for epoch in range(hparams['epochs']):

        model.train()
        t_l = 0
        cnt = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            # for batch, sampled_batch in enumerate(train_dataloader):
            for sampled_batch in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}")
                cnt += 1
                optimizer.zero_grad()
                output = model(sampled_batch['features'].squeeze(), sampled_batch['edge_index'].squeeze()).squeeze()
                target = sampled_batch['targets'].squeeze()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                t_l += loss.item()

            train_loss.append(t_l / cnt)

        model.eval()
        v_l = 0
        recall = 0
        precision = 0
        accuracy = 0
        cnt = 0
        with tqdm(val_dataloader, unit="batch") as tepoch:
            # for batch, sampled_batch in enumerate(val_dataloader):
            for sampled_batch in tepoch:
                tepoch.set_description(f"Val Epoch {epoch}")
                cnt += 1
                output = model(sampled_batch['features'].squeeze(), sampled_batch['edge_index'].squeeze()).squeeze()
                target = sampled_batch['targets'].squeeze()
                loss = criterion(output, target)
                v_l += loss.item()

                prediction = torch.sigmoid(output)
                prediction = torch.where(prediction < 0.5, 0, 1)
                confusion_matrix_ = confusion_matrix(prediction.long(), target.long())

                if confusion_matrix_.shape == (1, 1):
                    recall += 1.
                    precision += 1.
                    accuracy += 1.
                else:
                    # recall = TP/(TP + FN)
                    recall += confusion_matrix_[0][0] / (confusion_matrix_[0][0] + confusion_matrix_[1][0])

                    # precision = TP/(TP + FP)
                    precision += confusion_matrix_[0][0] / (confusion_matrix_[0][0] + confusion_matrix_[0][1])

                    # accuracy = (TP + TN)/(TP + TN + FP + FN)
                    accuracy += (confusion_matrix_[0][0] + confusion_matrix_[1][1]) / \
                                (confusion_matrix_[0][0] + confusion_matrix_[0][1] +
                                 confusion_matrix_[1][0] + confusion_matrix_[1][1])

        vl = v_l / cnt
        if vl < maxi_loss and epoch % 10 == 0:
            maxi_loss = vl
            save_model(model, 'model_k_' + str(k_back) + '.pth')
        val_loss.append(vl)
        val_prec.append(precision / cnt)
        val_acc.append(accuracy / cnt)
        val_rec.append(recall / cnt)

    return train_loss, val_loss, val_acc, val_prec, val_rec


def test(k, patient_dict, n=30):
    mimicdataset = MimicDataset(patient_dict,
                                n=n,
                                k_forward=k,
                                k_backward=k,
                                k_self=1)

    num_edges = mimicdataset[0]['edge_index'].squeeze().shape[1]
    degree_per_node = k + k + 1
    num_features = mimicdataset[0]['features'].squeeze().shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MimicModel(num_edges, degree_per_node, num_features).to(device).double()

    saved_model_path = os.path.join(save_path, 'model_k_' + str(k) + '.pth')
    model.load_state_dict(torch.load(saved_model_path))

    test_dataloader = DataLoader(mimicdataset, batch_size=1)

    criterion = torch.nn.BCEWithLogitsLoss()

    test_loss = []
    test_prec = []
    test_acc = []
    test_rec = []

    model.eval()
    for batch, sampled_batch in enumerate(test_dataloader):
        output = model(sampled_batch['features'].squeeze(), sampled_batch['edge_index'].squeeze()).squeeze()
        target = sampled_batch['targets'].squeeze()
        loss = criterion(output, target)
        test_loss.append(loss.item())

        prediction = torch.sigmoid(output)
        prediction = torch.where(prediction < 0.5, 0, 1)
        confusion_matrix_ = confusion_matrix(prediction.long(), target.long())

        if confusion_matrix_.shape == (1, 1):
            test_rec.append(1.)
            test_prec.append(1.)
            test_acc.append(1.)
        else:
            # recall = TP/(TP + FN)
            test_rec.append(confusion_matrix_[0][0] / (confusion_matrix_[0][0] + confusion_matrix_[1][0]))

            # precision = TP/(TP + FP)
            test_prec.append(confusion_matrix_[0][0] / (confusion_matrix_[0][0] + confusion_matrix_[0][1]))

            # accuracy = (TP + TN)/(TP + TN + FP + FN)
            test_acc.append((confusion_matrix_[0][0] + confusion_matrix_[1][1]) / \
                            (confusion_matrix_[0][0] + confusion_matrix_[0][1] +
                             confusion_matrix_[1][0] + confusion_matrix_[1][1]))

    return test_loss, test_acc, test_prec, test_rec


def train_and_plot_for_k(k, patient_dict, n=30):
    train_loss, val_loss, val_acc, val_prec, val_rec = train(patient_dict=patient_dict, k_back=k, k_for=k)

    plt.figure()
    plt.plot(np.array(train_loss), 'r', label="train_loss")
    plt.plot(np.array(val_loss), 'b', label="val_loss")
    plt.legend(loc="upper right")
    plt.title("Train Val Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, 'Train_Val_Curve_k_' + str(k) + '.png'))
    # plt.show()

    plt.figure()
    plt.plot(np.array(val_acc), 'r', label="val_acc")
    plt.legend(loc="upper right")
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_path, 'Val_Acc_Curve_k_' + str(k) + '.png'))
    # plt.show()


def test_and_plot_for_k(k, patient_dict, n=30):
    test_loss, test_acc, test_prec, test_rec = test(k=k, patient_dict=patient_dict)

    plt.figure()
    plt.plot(np.array(test_loss), 'r', label="test_loss")
    plt.legend(loc="upper right")
    plt.title("Test Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, 'Test_Loss_Curve_k_' + str(k) + '.png'))
    # plt.show()

    plt.figure()
    plt.plot(np.array(test_acc), 'r', label="test_acc")
    plt.legend(loc="upper right")
    plt.title("Test Accuracy Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_path, 'Test_Acc_Curve_k_' + str(k) + '.png'))

    return k, np.array(test_acc).sum() / len(patient_dict), np.array(test_prec).sum() / len(patient_dict), \
           np.array(test_rec).sum() / len(patient_dict)


def initialize_dataset(path, returntype='numpy'):
    return read_mimic_dataset(path, returntype=returntype)


# do this only once => save time !!
def initialize_datasets(returntype='numpy'):
    train_patient_dictionary = initialize_dataset(path_loc, returntype=returntype)
    test_patient_dictionary = initialize_dataset(test_path, returntype=returntype)
    return train_patient_dictionary, test_patient_dictionary


def conduct_experiment(k, train_patient_dictionary, test_patient_dictionary, n=30):
    train_and_plot_for_k(k=k, patient_dict=train_patient_dictionary, n=n)
    k, acc, prec, recall = test_and_plot_for_k(k=k, patient_dict=test_patient_dictionary, n=n)

    with open(os.path.join(save_path, 'experiment_results.txt'), 'a') as f:
        f.write(f"N={n}, K={k}, Accuracy={acc}, Precision={prec}, Recall={recall}, "
                f"F-Score={2*prec*recall/(prec + recall)}\n")
    f.close()


def getSmallerDataset(use_corr=False, type='NA'):
    '''
    Smaller Dataset refers to patients with only mortality_1 and same no. of patients with mortality_0
    '''

    print("For Test Data we want high accuracy, high precision, high recall. Often precision and recall cannot be"
          "high together. Thus we also calculate F-Score which can take highest values of 1 when precision and recall"
          "both are 1 (again not possible often)\n So we conduct experiments for different k's, keeping N=30")

    if use_corr:
        complete_patient_dictionary, test_patient_dictionary = select_features_corr(threshold=0.5, include_type=type)
    else:
        complete_patient_dictionary, test_patient_dictionary = initialize_datasets()

    patient_indices_mortality_only_0, patient_indices_mortality_only_1 = compareMortality(complete_patient_dictionary)

    # obviously len of patient_indices_mortality_only_1 <<< patient_indices_mortality_only_0
    n_1 = len(patient_indices_mortality_only_1)
    n_0 = len(patient_indices_mortality_only_0)
    print("\nObviously len of patient_indices_mortality_only_1 <<< patient_indices_mortality_only_0:")
    print(f"Count of Patients with only mortality 0 = {n_0}")
    print(f"Count of Patients with only mortality 1 = {n_1}")
    print("")

    # generate n_1 random numbers in range(0, n_0) which can be selected from patient_dict_with_mortality_only_0
    l = random.sample(range(n_0), n_1)

    # create a new dict for dataset
    train_patient_dict = {}
    cnt = 0
    for i in range(n_1):
        index_0 = patient_indices_mortality_only_0[l[i]]
        train_patient_dict[cnt] = complete_patient_dictionary[index_0]

        cnt += 1

        index_1 = patient_indices_mortality_only_1[i]
        train_patient_dict[cnt] = complete_patient_dictionary[index_1]

        cnt += 1


    # Now this is the complete train dict
    print(f"Length of train_dict = {len(train_patient_dict)}")
    print(f"Following patients used for training:")
    for p_index, patient in train_patient_dict.items():
        print(f"Patient Index:{p_index}, Patient Id:{patient['patient_id']}")

    # even places train_patient_dict[0,2,4...] = patients with mortality_0
    # odd places train_patient_dict[1,3,5...] = patients with mortality_1
    return train_patient_dict, test_patient_dictionary


def train_and_test(use_corr=False, type='NA'):
    train_patient_dict, test_patient_dictionary = getSmallerDataset(use_corr, type)

    # train the model for k=2 to 10
    for k in range(2, 10):
        conduct_experiment(k, train_patient_dict, test_patient_dictionary, n=30)


def save_or_show_graph(graph, save_or_show='save', save_title=''):
    """ Visualize the graph """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True)
    if save_or_show == 'show':
        plt.show()
        plt.close()
    else:
        plt.savefig(os.path.join(save_path, save_title + '.png'))
        plt.close()


def create_graphs_from_trained_model(k, n=30):

    edge_indices = []
    e1 = []
    e2 = []
    for i in range(n):
        for j in range(1, k + 1):
            if i - j >= 0:
                e1.append(i)
                e2.append(i - j)
        for j in range(1, k + 1):
            if i + j < n:
                e1.append(i)
                e2.append(i + j)
    edge_indices.append(e1)
    edge_indices.append(e2)

    num_edges = len(e1)
    degree_per_node = k + k
    num_features = 56 + 16 + 6

    # load saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MimicModel(num_edges, degree_per_node, num_features).to(device).double()
    saved_model_path = os.path.join(save_path, 'model_k_' + str(k) + '.pth')
    model.load_state_dict(torch.load(saved_model_path))

    G = nx.Graph()
    for i in range(num_edges):
        G.add_edge(e1[i], e2[i], weight=-model.edge_weight[i].detach())

    # use graphviz_layout to show heirarchical graphs
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True)
    save_or_show_graph(G, save_or_show='save', save_title='Prepared_Graph_k_' + str(k) + 'negative_wts')

    mst = nx.minimum_spanning_tree(G)
    pos = graphviz_layout(mst, prog='dot')
    nx.draw(mst, pos, with_labels=True)
    save_or_show_graph(mst, save_or_show='save', save_title='MST_k_' + str(k) + 'negative_wts')


def train_for_single_patient(train_patient_dict, k, n=30):

    edge_indices = []
    e1 = []
    e2 = []
    for i in range(n):
        for j in range(1, k + 1):
            if i - j >= 0:
                e1.append(i)
                e2.append(i - j)
        for j in range(1, k + 1):
            if i + j < n:
                e1.append(i)
                e2.append(i + j)
    edge_indices.append(e1)
    edge_indices.append(e2)

    # training begins:

    hparams = {'batch_size': 1, 'lr': 0.0001, 'epochs': 500, 'k_backward': k, 'k_forward': k, 'k_self': 1,
               'n': n, 'train_val_split': 0.80, 'wt_decay': 0.01}

    maxi_loss = 99999999

    patient_dict = train_patient_dict

    # dataset_size should be 3
    dataset_size = len(patient_dict)

    # ideally this should be same as the one printed while training --> check
    p_id = patient_dict[0]['patient_id']

    indices = list(range(dataset_size))
    split = 1
    train_indices, val_indices = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    mimicdataset = MimicDataset(patient_dict,
                                n=hparams['n'],
                                k_forward=hparams['k_forward'],
                                k_backward=hparams['k_backward'],
                                k_self=hparams['k_self'])

    num_edges = mimicdataset[0]['edge_index'].squeeze().shape[1]
    degree_per_node = hparams['k_backward'] + hparams['k_forward'] + hparams['k_self'] # doesn't matter for now
    num_features = mimicdataset[0]['features'].squeeze().shape[1]

    # initialize model from scratch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MimicModel(num_edges, degree_per_node, num_features).to(device).double()

    # uncomment to load saved model
    # saved_model_path = os.path.join(save_path, 'model_k_' + str(k) + '.pth')
    # model.load_state_dict(torch.load(saved_model_path))

    train_dataloader = DataLoader(mimicdataset, batch_size=hparams['batch_size'], sampler=train_sampler)
    val_dataloader = DataLoader(mimicdataset, batch_size=hparams['batch_size'], sampler=val_sampler)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['wt_decay'])

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loss = []
    val_loss = []

    for epoch in range(hparams['epochs']):

        model.train()
        t_l = 0
        cnt = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            # for batch, sampled_batch in enumerate(train_dataloader):
            for sampled_batch in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}, Training for patient: {sampled_batch['patient_id']}")
                cnt += 1
                optimizer.zero_grad()
                output = model(sampled_batch['features'].squeeze(), sampled_batch['edge_index'].squeeze()).squeeze()
                target = sampled_batch['targets'].squeeze()
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                t_l += loss.item()

            train_loss.append(t_l / cnt)

        model.eval()
        v_l = 0
        cnt = 0
        with tqdm(val_dataloader, unit="batch") as tepoch:
            # for batch, sampled_batch in enumerate(val_dataloader):
            for sampled_batch in tepoch:
                tepoch.set_description(f"Val Epoch {epoch}")
                cnt += 1
                output = model(sampled_batch['features'].squeeze(), sampled_batch['edge_index'].squeeze()).squeeze()
                target = sampled_batch['targets'].squeeze()
                loss = criterion(output, target)
                v_l += loss.item()

        vl = v_l / cnt
        if vl < maxi_loss and epoch % 10 == 0:
            maxi_loss = vl
            save_model(model, 'model_k_' + str(k) + '_p_id_' + str(p_id) + '.pth')
        val_loss.append(vl)

    return train_loss, val_loss


def train_and_plot_for_single_patient(train_patient_dict, k, n=30):
    p_id = train_patient_dict[0]['patient_id']
    train_loss, val_loss = train_for_single_patient(train_patient_dict, k, n=n)
    plt.figure()
    plt.plot(np.array(train_loss), 'r', label="train_loss")
    plt.plot(np.array(val_loss), 'b', label="val_loss")
    plt.legend(loc="upper right")
    plt.title("Train Val Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, 'Train_Val_Curve_k_' + str(k) + '_p_id_' + str(p_id) + '.png'))
    # plt.show()


def run_train_single_patient(k, train_patient_dict):
    '''
    For any 5 random paitents, train for them and create their graphs and MST's
    Logic:
    new_dict[0] = train_patient_dict[0] ==> Train <-- this is the main patient for which graph is made
    new_dict[1] = train_patient_dict[1] ==> Val
    new_dict[2] = train_patient_dict[2] ==> Val
    '''

    patient_list = []

    # create new dict out of existing dict with size 3 => 1 for training and 2 for val
    new_dict = {}
    new_dict[0] = train_patient_dict[0]
    new_dict[1] = train_patient_dict[1]
    new_dict[2] = train_patient_dict[2]
    train_and_plot_for_single_patient(train_patient_dict=new_dict, k=k, n=30)
    patient_list.append(new_dict[0]['patient_id'])

    # create new dict out of existing dict with size 3 => 1 for training and 2 for val
    new_dict = {}
    new_dict[0] = train_patient_dict[100]
    new_dict[1] = train_patient_dict[101]
    new_dict[2] = train_patient_dict[102]
    train_and_plot_for_single_patient(train_patient_dict=new_dict, k=k, n=30)
    patient_list.append(new_dict[0]['patient_id'])

    # create new dict out of existing dict with size 3 => 1 for training and 2 for val
    new_dict = {}
    new_dict[0] = train_patient_dict[401]
    new_dict[1] = train_patient_dict[402]
    new_dict[2] = train_patient_dict[403]
    train_and_plot_for_single_patient(train_patient_dict=new_dict, k=k, n=30)
    patient_list.append(new_dict[0]['patient_id'])

    # create new dict out of existing dict with size 3 => 1 for training and 2 for val
    new_dict = {}
    new_dict[0] = train_patient_dict[1005]
    new_dict[1] = train_patient_dict[1006]
    new_dict[2] = train_patient_dict[1007]
    train_and_plot_for_single_patient(train_patient_dict=new_dict, k=k, n=30)
    patient_list.append(new_dict[0]['patient_id'])

    # create new dict out of existing dict with size 3 => 1 for training and 2 for val
    new_dict = {}
    new_dict[0] = train_patient_dict[2303]
    new_dict[1] = train_patient_dict[2304]
    new_dict[2] = train_patient_dict[2305]
    train_and_plot_for_single_patient(train_patient_dict=new_dict, k=k, n=30)
    patient_list.append(new_dict[0]['patient_id'])

    return patient_list


def create_graphs_from_trained_model_single_patients(num_features, k, n=30, path="", p_id=None):

    edge_indices = []
    e1 = []
    e2 = []
    for i in range(n):
        for j in range(1, k + 1):
            if i - j >= 0:
                e1.append(i)
                e2.append(i - j)
        for j in range(1, k + 1):
            if i + j < n:
                e1.append(i)
                e2.append(i + j)
    edge_indices.append(e1)
    edge_indices.append(e2)

    num_edges = len(e1)
    degree_per_node = k + k

    # load saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MimicModel(num_edges, degree_per_node, num_features).to(device).double()
    saved_model_path = os.path.join(save_path, path)
    model.load_state_dict(torch.load(saved_model_path))

    G = nx.Graph()
    for i in range(num_edges):
        G.add_edge(e1[i], e2[i], weight=-model.edge_weight[i].detach())

    # use graphviz_layout to show heirarchical graphs
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True)
    save_or_show_graph(G, save_or_show='save', save_title='Prepared_Graph_k_' + str(k) + '_p_id' + str(p_id) + 'negative_wts')

    mst = nx.minimum_spanning_tree(G)
    pos = graphviz_layout(mst, prog='dot')
    nx.draw(mst, pos, with_labels=True)
    save_or_show_graph(mst, save_or_show='save', save_title='MST_k_' + str(k) + '_p_id' + str(p_id) + 'negative_wts')


def conduct_experiment_small_dicts(use_corr=False, type='NA'):
    # remember even for mortality_0 and odd for mortality_1
    small_patient_dict, _ = getSmallerDataset(use_corr, type)

    files = ["ts_vals_linear_imputed.csv",
             "ts_treatment.csv",
             "statics.csv",
             "rolling_tasks_binary_multilabel.csv"
             ]

    num_features = small_patient_dict[0][files[0]].shape[1] + small_patient_dict[0][files[1]].shape[1] \
                   + small_patient_dict[0][files[2]].shape[1]

    # patient list returned should be same for all the following methods
    patient_list = run_train_single_patient(k=2, train_patient_dict=small_patient_dict)
    run_train_single_patient(k=3, train_patient_dict=small_patient_dict)
    run_train_single_patient(k=4, train_patient_dict=small_patient_dict)
    run_train_single_patient(k=5, train_patient_dict=small_patient_dict)

    return num_features, patient_list


def create_graphs_from_trained_model_small(num_features, patient_list, k):
    p_id_list = patient_list
    for p_id in p_id_list:
        path = 'model_k_' + str(k) + '_p_id_' + str(p_id) + '.pth'
        create_graphs_from_trained_model_single_patients(num_features=num_features, k=k, n=30, path=path, p_id=p_id)


def select_features_corr(threshold=0.5, include_type='greater'):
    '''
    include_type = greater means select features with corr btw [-1,-threshold] and [threshold,+1] ; smaller means select features with corr btw [-threshold,threshold]
    '''

    complete_patient_dictionary, test_patient_dictionary = initialize_datasets(returntype='df')

    files = ["ts_vals_linear_imputed.csv",
             "ts_treatment.csv",
             "statics.csv",
             "rolling_tasks_binary_multilabel.csv"
             ]

    # remove cols from ts_vals_linear_imputed.csv that violate greater or smaller condition
    cols_to_remove_ts_vals = []

    # repeat for is_treatment.csv
    cols_to_remove_ts_treatment = []

    # <----- patients that did not die have all zeros therefore corr will be NAN instead corr is valid only for patients that died !!
    for k in complete_patient_dictionary.keys():

        df0 = complete_patient_dictionary[k][files[0]]
        df1 = complete_patient_dictionary[k][files[1]]
        df3 = complete_patient_dictionary[k][files[3]]

        std = df3['mort_24h'].std()

        if std > 0:
            for col in df0:
                corr = df3['mort_24h'].corr(df0[col])

                if math.isnan(corr):
                    cols_to_remove_ts_vals.append(col)
                else:
                    if include_type == 'greater':
                        if -threshold < corr < threshold:
                            cols_to_remove_ts_vals.append(col)
                    elif include_type == 'smaller':
                        if corr < -threshold or corr > threshold:
                            cols_to_remove_ts_vals.append(col)
                    else:
                        raise ValueError('include_type is not VALID ; possible values = [greater, smaller]')

            #df0.drop(columns=cols_to_remove, axis=1, inplace=True)
            #complete_patient_dictionary[k][files[0]] = df0

            for col in df1:
                corr = df3['mort_24h'].corr(df1[col])

                if math.isnan(corr):
                    cols_to_remove_ts_treatment.append(col)
                else:
                    if include_type == 'greater':
                        if -threshold < corr < threshold:
                            cols_to_remove_ts_treatment.append(col)
                    elif include_type == 'smaller':
                        if corr < -threshold or corr > threshold:
                            cols_to_remove_ts_treatment.append(col)
                    else:
                        raise ValueError('include_type is not VALID ; possible values = [greater, smaller]')

            #df1.drop(columns=cols_to_remove, axis=1, inplace=True)
            #complete_patient_dictionary[k][files[1]] = df1

            # convert all df's to numpy
            '''complete_patient_dictionary[k][files[0]] = complete_patient_dictionary[k][files[0]].to_numpy()
            complete_patient_dictionary[k][files[1]] = complete_patient_dictionary[k][files[1]].to_numpy()
            complete_patient_dictionary[k][files[2]] = complete_patient_dictionary[k][files[2]].to_numpy()
            complete_patient_dictionary[k][files[3]] = complete_patient_dictionary[k][files[3]].to_numpy()
    
            print(complete_patient_dictionary[k])'''

    cols_to_remove_ts_vals = list(set(cols_to_remove_ts_vals))
    cols_to_remove_ts_treatment = list(set(cols_to_remove_ts_treatment))

    print("Removed Features (because of NAN or threshold):")
    print(f"n={len(cols_to_remove_ts_vals)}, cols_to_remove_ts_vals={cols_to_remove_ts_vals}")
    print(f"n={len(cols_to_remove_ts_treatment)}, cols_to_remove_ts_treatment={cols_to_remove_ts_treatment}")

    # remove the removalble_features from complete_patient_dictionary and test_patient_dictionary
    for k in complete_patient_dictionary.keys():
        # ts_val
        complete_patient_dictionary[k][files[0]].drop(columns=cols_to_remove_ts_vals, axis=1, inplace=True)

        std_train = complete_patient_dictionary[k][files[3]].std()

        # ts_treatment
        complete_patient_dictionary[k][files[1]].drop(columns=cols_to_remove_ts_treatment, axis=1, inplace=True)

        complete_patient_dictionary[k][files[0]] = complete_patient_dictionary[k][files[0]].to_numpy()
        complete_patient_dictionary[k][files[1]] = complete_patient_dictionary[k][files[1]].to_numpy()
        complete_patient_dictionary[k][files[2]] = complete_patient_dictionary[k][files[2]].to_numpy()
        complete_patient_dictionary[k][files[3]] = complete_patient_dictionary[k][files[3]].to_numpy()

        #print(f"std_train={std_train}")
        #print(complete_patient_dictionary[k])

    for k in test_patient_dictionary.keys():
        # ts_val
        test_patient_dictionary[k][files[0]].drop(columns=cols_to_remove_ts_vals, axis=1, inplace=True)

        std_test = test_patient_dictionary[k][files[3]].std()

        # ts_treatment
        test_patient_dictionary[k][files[1]].drop(columns=cols_to_remove_ts_treatment, axis=1, inplace=True)

        test_patient_dictionary[k][files[0]] = test_patient_dictionary[k][files[0]].to_numpy()
        test_patient_dictionary[k][files[1]] = test_patient_dictionary[k][files[1]].to_numpy()
        test_patient_dictionary[k][files[2]] = test_patient_dictionary[k][files[2]].to_numpy()
        test_patient_dictionary[k][files[3]] = test_patient_dictionary[k][files[3]].to_numpy()

        #print(f"std_test={std_test}")
        #print(test_patient_dictionary[k])

    return complete_patient_dictionary, test_patient_dictionary





###################################### FUNCTIONS ENDS ##################################################################


#select_features_corr()
#select_features_corr(0.5, 'smaller')

#train_and_test()
'''
for k in range(2, 10):
    create_graphs_from_trained_model(k=k, n=30)
'''

num_features, patient_list = conduct_experiment_small_dicts(use_corr=True, type='smaller')
#'''
for k in range(2, 6):
    create_graphs_from_trained_model_small(num_features, patient_list, k)
#'''

