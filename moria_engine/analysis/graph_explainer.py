"""
Functions to explain the relationships in a graph.
"""

import networkx as nx
import pandas as pd 
from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema

def graph_centrality_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explain the relationships in a graph based on the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns.

    Returns:
        pd.DataFrame: A DataFrame containing explanations of relationships.
    """
    # Ensure required columns are present
    if not set(["table_name", "field_name", "domain_name"]).issubset(df.columns):
        raise ValueError("The DataFrame must contain 'table_name', 'field_name', and 'domain_name' columns.")

  
    # Build the adjacency matrix and table names - improve later to remove G0
    matrix, table_names, table_to_domain, G = build_common_fields_matrix_schema(df)

    # Calculate centrality measures
    centrality = nx.degree_centrality(G)

    # Calculate betweenness centrality
    betweeness_centrality = nx.betweenness_centrality(G)

    # Calculate closeness centrality
    closeness_centrality = nx.closeness_centrality(G)
    # Calculate eigenvector centrality
    eigenvector_centrality = nx.eigenvector_centrality(G)
    # Calculate PageRank
    pagerank = nx.pagerank(G)
    # Calculate harmonic centrality
    harmonic_centrality = nx.harmonic_centrality(G)
    # Calculate Katz centrality
    katz_centrality = nx.katz_centrality(G)    


    # Create a DataFrame to hold explanations
    attributes = pd.DataFrame({
        'node': table_names,
        'domain_name': [table_to_domain[name] for name in table_names],
        'degree': [item[1] for item in G.degree()],
        'centrality': list(centrality.values()),
        'betweenness_centrality': list(betweeness_centrality.values()),
        #'closeness_centrality': list(closeness_centrality.values()),
        #'eigenvector_centrality': list(eigenvector_centrality.values()),
        #'pagerank': list(pagerank.values()),
        #'harmonic_centrality': list(harmonic_centrality.values()),
        #'katz_centrality': list(katz_centrality.values())  
    })

    return attributes 

def graph_centrality_narrative(df: pd.DataFrame) -> str:
    """
    Generate a narrative explanation of the relationships in a graph based on centrality measures.

    Args:
        df (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns.

    Returns:
        str: A narrative explanation of the relationships.
    """
    attributes = graph_centrality_attributes(df)
    
    # Generate narrative
    narrative = "Graph Centrality Analysis:\n\n"
    
    for _, row in attributes.iterrows():
        narrative += f"Node '{row['node']}' in domain '{row['domain_name']}' has:\n"
        narrative += f"- Degree: {row['degree']}\n"
        narrative += f"- Centrality: {row['centrality']:.4f}\n"
        narrative += f"- Betweenness Centrality: {row['betweenness_centrality']:.4f}\n"
        narrative += "\n"

    return narrative.strip() 

def graph_centrality_summary_warning(df: pd.DataFrame) -> str:
    """
    Generate a summary warning for the graph centrality analysis.

    Args:
        df (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns.

    Returns:
        str: A summary warning message.
    """
    attributes = graph_centrality_attributes(df)
    
    # Generate summary warning
    summary = "Graph Centrality Summary Warning:\n\n"
    summary += "\n\nThis analysis helps identify key nodes in the graph that may have significant influence on the overall structure and relationships."
    summary += "\n\n"
    summary += "The following nodes have been analyzed based on their centrality measures:\n\n" 
    if attributes.empty:
        summary += "No nodes found in the graph. Please check the input data."
    else:
        high_centrality_nodes = attributes[attributes['centrality'] > 0.5]
        if not high_centrality_nodes.empty:
            summary += "Warning: The following nodes have high centrality values, indicating potential key roles in the graph. High impact to the whole network if there is a failure or severe quality issues in the node:\n"
            for _, row in high_centrality_nodes.iterrows():
                summary += f"- Node '{row['node']}' in domain '{row['domain_name']}' with Centrality: {row['centrality']:.4f}\n"
        else:
            summary += "No nodes with high centrality values found."
        summary += "\n\n"
        low_centrality_nodes = attributes[attributes['betweenness_centrality'] < 0.05]
        if not low_centrality_nodes.empty:
            summary += "Warning: The following nodes have low betweenness centrality values, indicating potential isolation/low data quality. Low impact to the whole network if there is a failure or severe quality issues in the node:\n"
            for _, row in low_centrality_nodes.iterrows():
                summary += f"- Node '{row['node']}' in domain '{row['domain_name']}' with Betweenness Centrality: {row['betweenness_centrality']:.4f}\n"
        else:
            summary += "No nodes with low centrality values found."

    summary += "\nPlease review the graph for potential key nodes and their relationships."
    
    return summary.strip()

def graph_centrality_attributes_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explain the relationships in a graph based on the input DataFrame and KPIs.

    Args:
        df (pd.DataFrame): Input DataFrame with 'kpi_name', 'table_name', and 'domain_name' columns.

    Returns:
        pd.DataFrame: A DataFrame containing explanations of relationships, including centrality measures.
    """
    # Ensure required columns are present
    if not set(["kpi_name", "table_name", "domain_name"]).issubset(df.columns):
        raise ValueError("The DataFrame must contain 'kpi_name', 'table_name', and 'domain_name' columns.")

    # Build the adjacency matrix and mappings using build_common_fields_matrix_kpis
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_kpis
    matrix, table_names, table_to_domain, table_to_kpi = build_common_fields_matrix_kpis(df)

    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(matrix, create_using=nx.Graph())
    nx.set_node_attributes(G, table_to_domain, "domain_name")
    nx.set_node_attributes(G, table_to_kpi, "kpi_name")

    # Calculate centrality measures
    centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Create a DataFrame to hold explanations
    attributes = pd.DataFrame({
        'node': table_names,
        'domain_name': [table_to_domain[name] for name in table_names],
        'kpi_name': [", ".join(table_to_kpi[name]) for name in table_names],  # Join multiple KPIs
        'degree': [item[1] for item in G.degree()],
        'centrality': list(centrality.values()),
        'betweenness_centrality': list(betweenness_centrality.values()),
        'closeness_centrality': list(closeness_centrality.values())
    })

    return attributes



def graph_edit_distance_between_schema_and_kpis(G_schema: nx.Graph, G_kpis: nx.Graph, timeout: float = 5.0) -> float:
    """
    Compute the graph edit distance between the schema graph and the KPIs graph.

    Args:
        G_schema (nx.Graph): The schema graph.
        G_kpis (nx.Graph): The KPIs graph.
        timeout (float): Maximum time in seconds to compute the distance (default: 5.0).

    Returns:
        float: The graph edit distance (lower means more similar).
    """
    try:
        # networkx.graph_edit_distance returns a generator, so we take the first value <- Nope
        # This is a blocking operation, so we set a timeout
        ged_iter = nx.algorithms.similarity.graph_edit_distance(G_schema, G_kpis, timeout=timeout)
        distance = ged_iter
        return distance
    except Exception as e:
        raise RuntimeError(f"Error computing graph edit distance: {e}")
