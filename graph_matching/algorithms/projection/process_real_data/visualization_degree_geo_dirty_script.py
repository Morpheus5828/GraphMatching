import numpy as np
import os
import sys
import pickle as p
import networkx as nx
import random
import plotly.figure_factory as ff

current_dir = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_dir, '../../..'))
if project_path not in sys.path:
    sys.path.append(project_path)

path_1 = os.path.join(project_path, 'data/_obsolete_OASIS_full_batch/modified_graphs/')

# Generate random color codes for plotting
def generate_random_color_codes(num_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(num_colors)]
    return color

# Function to remove dummy nodes from the graph
def remove_dummy_nodes(graph):
    G = graph.copy()
    to_remove = [p for p, d in G.nodes(data=True) if d.get('is_dummy', False) == True]
    G.remove_nodes_from(to_remove)
    return G

# Read graphs and compute degree distribution
degree_list = []
for graph in os.listdir(path_1):
    with open(os.path.join(path_1, graph), "rb") as f:
        G = p.load(f)
    new_G = remove_dummy_nodes(G)
    new_G.remove_edges_from(nx.selfloop_edges(new_G))  # Remove self loops
    degree_list.append(list(dict(nx.degree(new_G)).values()))

# Plot degree distribution
color = generate_random_color_codes(len(degree_list))
fig = ff.create_distplot(degree_list, os.listdir(path_1), show_hist=False, colors=color)
fig.update_layout(title_text='Degree density real graph')
fig.show()

def degree_of_neighb(G):
    nb_degree_dict = {node: [G.degree(nb) for nb in G.neighbors(node)] for node in G}
    return nb_degree_dict

all_graph_nb_degree = []
for graph in os.listdir(path_1):
    with open(os.path.join(path_1, graph), "rb") as f:
        graph = p.load(f)
    graph = remove_dummy_nodes(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))  # Remove self loops
    all_graph_nb_degree.append(degree_of_neighb(graph))

avg_nb_degree_all_graph = [[np.mean(lst) for lst in degree_dict.values()] for degree_dict in all_graph_nb_degree]

# Plot average neighbor degree distribution
color = generate_random_color_codes(len(avg_nb_degree_all_graph))
fig = ff.create_distplot(avg_nb_degree_all_graph, os.listdir(path_1), show_hist=False, bin_size=.2, colors=color)
fig.update_layout(title_text='Average neighbor degree for real graphs')
fig.show()

# Function to compute geodesic distance of neighbors for each node in the graph
def geo_dist_of_neighb(G):
    nb_dist_dict = {node: [G.get_edge_data(node, nb)['geodesic_distance'] for nb in G.neighbors(node)] for node in G}
    return nb_dist_dict

# Compute average neighbor geodesic distance for each graph
all_graph_nb_distance = []
for graph in os.listdir(path_1):
    with open(os.path.join(path_1, graph), "rb") as f:
        graph = p.load(f)
    graph = remove_dummy_nodes(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))  # Remove self loops
    all_graph_nb_distance.append(geo_dist_of_neighb(graph))

avg_nb_dist_all_graph = [[np.mean(lst) for lst in dist_dict.values()] for dist_dict in all_graph_nb_distance]

# Plot average neighbor geodesic distance distribution
color = generate_random_color_codes(len(avg_nb_dist_all_graph))
fig = ff.create_distplot(avg_nb_dist_all_graph, os.listdir(path_1), show_hist=False, bin_size=.2, colors=color)
fig.update_layout(title_text='Average neighbor geo distance for real graphs')
fig.show()
