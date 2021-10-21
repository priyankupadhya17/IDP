import networkx as nx
import matplotlib.pyplot as plt
import random


def gen_graph(n, m):
    graph = nx.gnm_random_graph(n, m)
    max_weight = 100

    for (u, v, w) in graph.edges(data=True):
        w['weight'] = random.randint(1, max_weight)

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


def create_graph_dataset(n):
    #filename = "graph_dataset.txt"
    filename = None

    max_nodes = 100
    min_nodes = 5

    for i in range(n):

        n_nodes = random.randint(min_nodes, max_nodes)

        max_edges = int(n_nodes * (n_nodes - 1) / 2)
        min_edges = n_nodes

        n_edges = random.randint(min_edges, max_edges)

        graph = gen_graph(n_nodes, n_edges)

        mst = nx.minimum_spanning_tree(graph)

        print(sorted(mst.edges(data=True)))

        #show_graph(graph)
        '''
        if i % 3 == 0:
            show_graph(graph)
        '''
        output_edge_list(graph, filename)
