"""Descriptive networks of thresholded candidates, without gene/cluster p-values."""
import networkx as nx
import pandas as pd


def candidate_network(A, nodes, candidates):
    ids = sorted(set(candidates.entrez.astype(int))) if len(candidates) else []
    pos = {int(g): i for i, g in enumerate(nodes)}
    ids = [g for g in ids if g in pos]
    sub = A[[pos[g] for g in ids]][:, [pos[g] for g in ids]].tocoo()
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_edges_from((ids[i], ids[j]) for i, j in zip(sub.row, sub.col) if i < j)
    assignment = {}
    community = 0
    for component in sorted(nx.connected_components(graph), key=lambda c: (-len(c), min(c))):
        g = graph.subgraph(component)
        groups = nx.community.greedy_modularity_communities(g) if g.number_of_edges() else [component]
        for group in groups:
            community += 1
            assignment.update({node: community for node in group})
    table = pd.DataFrame({"entrez": ids, "community": [assignment[g] for g in ids],
        "degree_in_candidate_network": [graph.degree(g) for g in ids],
        "degree_in_background": [float(A[pos[g]].sum()) for g in ids]})
    table["interpretation"] = "descriptive_partition_no_community_significance"
    return graph, table
