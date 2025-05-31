import networkx as nx
import numpy as np

def random_walker_shannon_entropy(G: nx.Graph) -> float:
    """
    Calculate the Random Walker Shannon Entropy for a given network.

    The metric is defined as the sum of the logarithm of the degrees of the vertices
    with non-zero degrees in the network.

    Args:
        G (nx.Graph): A NetworkX graph.

    Returns:
        float: The Random Walker Shannon Entropy of the network.

    Refence:
        https://en.wikipedia.org/wiki/Network_entropy
    """
    # Get the degree of each node
    degrees = dict(G.degree())
    
    # Filter out nodes with zero degree
    non_zero_degrees = [deg for deg in degrees.values() if deg > 0]
    
    # Calculate the entropy
    entropy = sum(np.log(non_zero_degrees))/(len(degrees)* np.log(len(degrees)-1)) if non_zero_degrees else 0.0
    # Normalize the entropy
    
    return entropy

def random_walker_shannon_entropy_with_missing_link(G: nx.Graph, u: str, v: str) -> float:
    """
    Calculate the Random Walker Shannon Entropy of the network after adding a missing link.

    Args:
        G (nx.Graph): A NetworkX graph.
        u (str): The first node of the missing link.
        v (str): The second node of the missing link.

    Returns:
        float: The Random Walker Shannon Entropy of the network after adding the link.
    """
    # Temporarily add the missing link
    G.add_edge(u, v)
    
    # Compute the entropy with the added link
    entropy = random_walker_shannon_entropy(G)
    
    # Remove the link to restore the original graph
    G.remove_edge(u, v)
    
    return entropy

def find_max_entropy_shift_link(G: nx.Graph) -> tuple[str, str, float]:
    """
    Find the missing link that maximizes the entropy shift when added to the network.

    Args:
        G (nx.Graph): A NetworkX graph with table names as node labels.

    Returns:
        tuple: A tuple containing the table names of the missing link (u, v) and the maximum entropy shift.
    """
    # Get all possible pairs of nodes in the graph
    nodes = list(G.nodes)
    possible_links = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:] if not G.has_edge(u, v)]

    max_entropy_shift = float('-inf')
    best_link = None

    # Compute the entropy shift for each possible missing link
    original_entropy = random_walker_shannon_entropy(G)
    for u, v in possible_links:
        # Temporarily add the link and compute the entropy
        entropy_with_link = random_walker_shannon_entropy_with_missing_link(G, u, v)
        
        # Calculate the entropy shift
        entropy_shift = entropy_with_link - original_entropy
        
        # Update the best link if this shift is the largest
        if entropy_shift > max_entropy_shift:
            max_entropy_shift = entropy_shift
            best_link = (u, v)

    return best_link[0], best_link[1], max_entropy_shift

def betweenness_centrality_with_missing_node(G: nx.Graph, node: str) -> dict:
    """
    Calculate the betweenness centrality of the network after removing a specific node.

    Args:
        G (nx.Graph): A NetworkX graph.
        node (str): The node to be temporarily removed.

    Returns:
        dict: A dictionary where keys are node names and values are their betweenness centrality
              after the specified node is removed.
    """
    if node not in G.nodes:
        raise ValueError(f"Node '{node}' is not in the graph.")

    # Temporarily remove the node
    G.remove_node(node)
    
    # Compute the betweenness centrality for the modified graph
    centrality = nx.betweenness_centrality(G)
    
    # Restore the original graph
    G.add_node(node)
    
    return centrality

def find_max_betweenness_shift_node(G: nx.Graph) -> tuple[str, float]:
    """
    Find the node (table name) that causes the largest change in total betweenness centrality
    when removed from the network.

    Args:
        G (nx.Graph): A NetworkX graph with table names as node labels.

    Returns:
        tuple: A tuple containing the table name of the node and the maximum betweenness shift.
    """
    max_betweenness_shift = float('-inf')
    best_node = None

    # Compute the total betweenness centrality of the original graph
    original_centrality = nx.betweenness_centrality(G)
    original_total_centrality = sum(original_centrality.values())

    # Iterate over all nodes in the graph
    for node in G.nodes:
        G0 = G.copy()  # Create a copy of the graph to avoid modifying the original
        # Compute the betweenness centrality after removing the node
        centrality_after_removal = betweenness_centrality_with_missing_node(G0, node)
        total_centrality_after_removal = sum(centrality_after_removal.values())

        # Calculate the betweenness shift
        betweenness_shift = original_total_centrality - total_centrality_after_removal

        # Update the best node if this shift is the largest
        if betweenness_shift > max_betweenness_shift:
            max_betweenness_shift = betweenness_shift
            best_node = node

    return best_node, max_betweenness_shift