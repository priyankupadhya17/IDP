import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


def gen_graph(n, m):
    graph = nx.gnm_random_graph(n, m)
    #max_weight = 100

    for (u, v, w) in graph.edges(data=True):
        w['weight'] = np.random.uniform(low=0, high=1)

    return graph


def show_graph(graph):
    """ Visualize the graph """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True)
    plt.show()


def output_edge_list(graph, filename):

    if filename is None:
        print(graph.number_of_nodes(), graph.number_of_edges())
        for (u, v, w) in graph.edges(data=True):
            print(u, v, w['weight'])
    else:
        with open(filename, 'a') as fout:
            print(graph.number_of_nodes(), graph.number_of_edges(), file=fout)
            for (u, v, w) in graph.edges(data=True):
                print(u, v, w['weight'], file=fout)


def create_graph_dataset():
    #filename = "graph_dataset.txt"
    filename = None

    max_nodes = 6#100
    min_nodes = 6


    n_nodes = random.randint(min_nodes, max_nodes)

    max_edges = int(n_nodes * (n_nodes - 1) / 2)
    min_edges = n_nodes

    n_edges = 15#random.randint(min_edges, max_edges)

    graph = gen_graph(n_nodes, n_edges)

    mst = nx.minimum_spanning_tree(graph)

    line_graph = nx.line_graph(graph)

    #print(sorted(mst.edges(data=True)))
    #show_graph(graph)
    #show_graph(line_graph)
    
    line_graph_nodes = []
    
    line_graph_feature_dict = {}
    for node in line_graph.nodes():
        (u, v) = node
        line_graph_feature_dict[node] = {'features': graph[u][v]['weight']}
        line_graph_nodes.append(node)
    
    nx.set_node_attributes(line_graph, line_graph_feature_dict)
    
    return graph, line_graph, mst, line_graph_nodes
