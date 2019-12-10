import networkx as nx
import dwave_networkx as dnx
import dwave_networkx.drawing
import matplotlib.pyplot as plt


def draw_chimera_graph(embedding):

    plt.ion()
    G = dnx.chimera_graph(16,16,4)
    dnx.draw_chimera_embedding(G, embedding, show_labels=True)
    plt.show()