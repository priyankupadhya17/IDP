import networkx as nx
import matplotlib.pyplot as plt
import random


def gen_graph(n, m):
    graph = nx.gnm_random_graph(n, m)
    return graph


def show_graph(graph):
    """ Visualize the graph """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True)
    plt.show()


def output_edge_list(graph, one_based, filename, max_weight):
    """ Output num node, num edge and edge list """
    num_node = graph.number_of_nodes()
    num_edge = graph.number_of_edges()
    base = 1 if one_based else 0
    edge_list = [[edge[0] + base, edge[1] + base] for edge in graph.edges()]

    if filename is None:
        print(num_node, num_edge)
        for edge in edge_list:
            print(edge[0], edge[1], random.randint(0, max_weight))
        return
    with open(filename, 'a') as fout:
        print(num_node, num_edge, file=fout)
        for edge in edge_list:
            print(edge[0], edge[1], random.randint(0, max_weight), file=fout)
    print('Saved edge list in %s' % filename)


def create_graph_dataset(n):
    filename = "graph_dataset.txt"

    max_nodes = 100
    min_nodes = 5

    for i in range(n):

        n_nodes = random.randint(min_nodes, max_nodes)

        max_edges = int(n_nodes * (n_nodes - 1) / 2)
        min_edges = n_nodes

        n_edges = random.randint(min_edges, max_edges)

        graph = gen_graph(n_nodes, n_edges)

        '''
        if i % 3 == 0:
            show_graph(graph)
        '''
        output_edge_list(graph, False, filename, 100)
