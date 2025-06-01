"""
Functions to transform dataframes of domains, tables, fields.
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pyvis.network import Network
from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema, build_common_fields_matrix_kpis
import matplotlib.patches as patches

def visualize_network_schema(df: pd.DataFrame):
    """
    Visualize the adjacency matrix as a network graph.

    Args:
        df (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns.
    """
    # Build the adjacency matrix and table names - improve later to remove G0
    matrix, table_names, table_to_domain, G0 = build_common_fields_matrix_schema(df)

    # Create a graph from the adjacency matrix
    G = nx.Graph()

    # Add nodes with domain_name as an attribute
    for table_name in table_names:
        G.add_node(table_name, domain=table_to_domain[table_name])

    # Add edges with weights (number of common fields)
    for i, table_i in enumerate(table_names):
        for j, table_j in enumerate(table_names):
            if matrix[i, j] > 0 and i!=j:  # Only add edges with non-zero weights and avoid self-loops
                G.add_edge(table_i, table_j, weight=matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Set colors for nodes based on centrality
    node_colors = list(nx.degree_centrality(G).values())

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=10, font_weight='bold',node_shape='h')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8)
    plt.title("Network Visualization of Common Fields")
    plt.show()

    return G

def visualize_network_interactive_schema(df: pd.DataFrame, output_file="network.html"):
    """
    Visualize the adjacency matrix as an interactive network graph using pyvis.

    Args:
        df (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns.
        output_file (str): Path to save the interactive HTML file.
    """
    # Build the adjacency matrix and table names
    matrix, table_names, table_to_domain = build_common_fields_matrix_schema(df)

    # Create a pyvis Network
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Create a graph from the adjacency matrix
    G = nx.Graph()

    # Add nodes with domain_name as an attribute
    for table_name in table_names:
        G.add_node(table_name, domain=table_to_domain[table_name])

    # Add edges with weights (number of common fields)
    for i, table_i in enumerate(table_names):
        for j, table_j in enumerate(table_names):
            if matrix[i, j] > 0:  # Only add edges with non-zero weights
                G.add_edge(table_i, table_j, weight=matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    edge_weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 8))
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.set_edge_smooth('dynamic')


    plt.title("Network Visualization of Common Fields")
    plt.show()
    # Generate and open the interactive visualization
    net.show(output_file)

def visualize_network_interactive_kpis(df: pd.DataFrame, output_file="network_kpis.html"):
    """
    Visualize the adjacency matrix as an interactive network graph using pyvis, based on KPIs.

    Args:
        df (pd.DataFrame): Input DataFrame with 'kpi_name', 'table_name', and 'domain_name' columns.
        output_file (str): Path to save the interactive HTML file.
    """
    

    # Build the adjacency matrix and table names using the KPI-based function
    matrix, table_names, table_to_domain, table_to_kpi = build_common_fields_matrix_kpis(df)

    # Create a pyvis Network
    net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes with domain_name and kpi_name as attributes
    for table_name in table_names:
        net.add_node(
            table_name,
            label=table_name,
            title=f"Domain: {table_to_domain[table_name]}<br>KPIs: {', '.join(table_to_kpi[table_name])}",
            color="#87CEEB"  # Light blue for nodes
        )

    # Add edges with weights (connections based on shared KPIs)
    for i, table_i in enumerate(table_names):
        for j, table_j in enumerate(table_names):
            if matrix[i, j] > 0:  # Only add edges with non-zero weights
                net.add_edge(
                    table_i,
                    table_j,
                    value=matrix[i, j],
                    title=f"Shared KPI Connection"
                )

    # Generate and open the interactive visualization
    net.show_buttons(filter_=['physics'])
    net.set_edge_smooth('dynamic')
    net.show(output_file)

def visualize_network_kpis(df: pd.DataFrame):
    """
    Visualize the adjacency matrix as a network graph based on KPIs.

    Args:
        df (pd.DataFrame): Input DataFrame with 'kpi_name', 'table_name', and 'domain_name' columns.
    """
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_kpis

    # Build the adjacency matrix and table names using the KPI-based function
    matrix, table_names, table_to_domain, table_to_kpi = build_common_fields_matrix_kpis(df)

    # Create a graph from the adjacency matrix
    G = nx.Graph()

    # Add nodes with domain_name and kpi_name as attributes
    for table_name in table_names:
        G.add_node(
            table_name,
            domain=table_to_domain[table_name],
            kpi_name=", ".join(table_to_kpi[table_name])  # Join multiple KPIs into a single string
        )

    # Add edges with weights (connections based on shared KPIs)
    for i, table_i in enumerate(table_names):
        for j, table_j in enumerate(table_names):
            if matrix[i, j] > 0 and i != j:  # Only add edges with non-zero weights and avoid self-loops
                G.add_edge(table_i, table_j, weight=matrix[i, j])

    # Draw the graph
    pos = nx.spring_layout(G)  # Layout for better visualization
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Set colors for nodes based on centrality
    node_colors = list(nx.degree_centrality(G).values())

    plt.figure(figsize=(10, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=10,
        font_weight='bold',
        node_shape='h'
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_size=8)
    plt.title("Network Visualization Based on KPIs")
    plt.show()

    return G

def visualize_combined_networks(df_schema: pd.DataFrame, df_kpis: pd.DataFrame):
    """
    Visualize two networks side by side: one based on schema relationships and the other based on KPIs.

    Args:
        df_schema (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns for schema relationships.
        df_kpis (pd.DataFrame): Input DataFrame with 'kpi_name', 'table_name', and 'domain_name' columns for KPI relationships.
    """
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema, build_common_fields_matrix_kpis

    # Build the schema-based network
    matrix_schema, table_names_schema, table_to_domain_schema, _ = build_common_fields_matrix_schema(df_schema)

    # Ensure all table names in schema are present in KPI DataFrame
    table_names_kpis = df_kpis["table_name"].unique().tolist()
    missing_tables = set(table_names_schema) - set(table_names_kpis)
    if missing_tables:
        # Add missing table names to df_kpis with domain_name from df_schema and kpi_name as the table name
        missing_rows = pd.DataFrame({
            "kpi_name": list(missing_tables),  # Use the table name as the KPI name
            "table_name": list(missing_tables),
            "domain_name": [table_to_domain_schema[table] for table in missing_tables]  # Get domain_name from df_schema
        })
        df_kpis = pd.concat([df_kpis, missing_rows], ignore_index=True)

    # Build the KPI-based network
    matrix_kpis, table_names_kpis, table_to_domain_kpis, table_to_kpi = build_common_fields_matrix_kpis(df_kpis)

    # Create the schema-based graph
    G_schema = nx.Graph()
    for table_name in table_names_schema:
        G_schema.add_node(table_name, domain=table_to_domain_schema[table_name])
    for i, table_i in enumerate(table_names_schema):
        for j, table_j in enumerate(table_names_schema):
            if matrix_schema[i, j] > 0 and i != j:
                G_schema.add_edge(table_i, table_j, weight=matrix_schema[i, j])

    # Create the KPI-based graph
    G_kpis = nx.Graph()
    for table_name in table_names_kpis:
        G_kpis.add_node(table_name, domain=table_to_domain_kpis.get(table_name), kpi_name=", ".join(table_to_kpi.get(table_name, [])))
    for i, table_i in enumerate(table_names_kpis):
        for j, table_j in enumerate(table_names_kpis):
            if matrix_kpis[i, j] > 0 and i != j:
                G_kpis.add_edge(table_i, table_j, weight=matrix_kpis[i, j])

    # Use the same layout for both graphs
    pos = nx.spring_layout(G_schema)  # Shared layout for consistent node positions

    # Plot the networks side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the schema-based network
    edge_weights_schema = nx.get_edge_attributes(G_schema, 'weight')
    node_colors_schema = list(nx.degree_centrality(G_schema).values())
    nx.draw(
        G_schema,
        pos,
        ax=axes[0],
        with_labels=True,
        node_color=node_colors_schema,
        node_size=2000,
        font_size=10,
        font_weight='bold',
        node_shape='h'
    )
    nx.draw_networkx_edge_labels(G_schema, pos, edge_labels=edge_weights_schema, ax=axes[0], font_size=8)
    axes[0].set_title("Schema-Based Network")

    # Plot the KPI-based network
    edge_weights_kpis = nx.get_edge_attributes(G_kpis, 'weight')
    node_colors_kpis = list(nx.degree_centrality(G_kpis).values())
    nx.draw(
        G_kpis,
        pos,
        ax=axes[1],
        with_labels=True,
        node_color=node_colors_kpis,
        node_size=2000,
        font_size=10,
        font_weight='bold',
        node_shape='h'
    )
    nx.draw_networkx_edge_labels(G_kpis, pos, edge_labels=edge_weights_kpis, ax=axes[1], font_size=8)
    axes[1].set_title("KPI-Based Network")

    # Show the combined plot
    plt.tight_layout()
    plt.show()

def visualize_combined_networks1(df_schema: pd.DataFrame, df_kpis: pd.DataFrame):
    """
    Visualize two networks side by side: one based on schema relationships and the other based on KPIs.

    Args:
        df_schema (pd.DataFrame): Input DataFrame with 'table_name', 'field_name', and 'domain_name' columns for schema relationships.
        df_kpis (pd.DataFrame): Input DataFrame with 'kpi_name', 'table_name', and 'domain_name' columns for KPI relationships.
    """
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema, build_common_fields_matrix_kpis

    # Ensure all table names in KPI DataFrame are present in schema DataFrame
    table_names_schema = df_schema["table_name"].unique().tolist()
    table_names_kpis = df_kpis["table_name"].unique().tolist()
    missing_tables_in_schema = set(table_names_kpis) - set(table_names_schema)
    if missing_tables_in_schema:
        # Add missing table names to df_schema with null values
        missing_rows = pd.DataFrame({
            "table_name": list(missing_tables_in_schema),
            "field_name": [None] * len(missing_tables_in_schema),
            "domain_name": [None] * len(missing_tables_in_schema)
        })
        df_schema = pd.concat([df_schema, missing_rows], ignore_index=True)

    # Ensure all table names in schema DataFrame are present in KPI DataFrame
    table_names_schema = df_schema["table_name"].unique().tolist()  # Update after adding missing rows
    missing_tables_in_kpis = set(table_names_schema) - set(table_names_kpis)
    if missing_tables_in_kpis:
        # Add missing table names to df_kpis with domain_name from df_schema and kpi_name as the table name
        table_to_domain_schema = df_schema.drop_duplicates(subset="table_name").set_index("table_name")["domain_name"].to_dict()
        missing_rows = pd.DataFrame({
            "kpi_name": list(missing_tables_in_kpis),  # Use the table name as the KPI name
            "table_name": list(missing_tables_in_kpis),
            "domain_name": [table_to_domain_schema.get(table) for table in missing_tables_in_kpis]
        })
        df_kpis = pd.concat([df_kpis, missing_rows], ignore_index=True)

    # Build the schema-based network
    matrix_schema, table_names_schema, table_to_domain_schema, _ = build_common_fields_matrix_schema(df_schema)

    # Build the KPI-based network
    matrix_kpis, table_names_kpis, table_to_domain_kpis, table_to_kpi = build_common_fields_matrix_kpis(df_kpis)

    # Create the schema-based graph
    G_schema = nx.Graph()
    for table_name in table_names_schema:
        G_schema.add_node(table_name, domain=table_to_domain_schema[table_name])
    for i, table_i in enumerate(table_names_schema):
        for j, table_j in enumerate(table_names_schema):
            if matrix_schema[i, j] > 0 and i != j:
                G_schema.add_edge(table_i, table_j, weight=matrix_schema[i, j])

    # Create the KPI-based graph
    G_kpis = nx.Graph()
    for table_name in table_names_kpis:
        G_kpis.add_node(table_name, domain=table_to_domain_kpis.get(table_name), kpi_name=", ".join(table_to_kpi.get(table_name, [])))
    for i, table_i in enumerate(table_names_kpis):
        for j, table_j in enumerate(table_names_kpis):
            if matrix_kpis[i, j] > 0 and i != j:
                G_kpis.add_edge(table_i, table_j, weight=matrix_kpis[i, j])

    # Use the same layout for both graphs
    pos = nx.spring_layout(G_schema)  # Shared layout for consistent node positions

    # Plot the networks side by side
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the schema-based network
    edge_weights_schema = nx.get_edge_attributes(G_schema, 'weight')
    node_colors_schema = list(nx.degree_centrality(G_schema).values())
    nx.draw(
        G_schema,
        pos,
        ax=axes[0],
        with_labels=True,
        node_color=node_colors_schema,
        node_size=2000,
        font_size=10,
        font_weight='bold',
        node_shape='h'
    )
    nx.draw_networkx_edge_labels(G_schema, pos, edge_labels=edge_weights_schema, ax=axes[0], font_size=8)
    axes[0].set_title("Schema-Based Network")

    # Add a rectangular box around the schema-based network
    rect = patches.Rectangle(
        (0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', transform=axes[0].transAxes
    )
    axes[0].add_patch(rect)

    # Plot the KPI-based network
    edge_weights_kpis = nx.get_edge_attributes(G_kpis, 'weight')
    node_colors_kpis = list(nx.degree_centrality(G_kpis).values())
    nx.draw(
        G_kpis,
        pos,
        ax=axes[1],
        with_labels=True,
        node_color=node_colors_kpis,
        node_size=2000,
        font_size=10,
        font_weight='bold',
        node_shape='h'
    )
    nx.draw_networkx_edge_labels(G_kpis, pos, edge_labels=edge_weights_kpis, ax=axes[1], font_size=8)
    axes[1].set_title("KPI-Based Network")

    # Add a rectangular box around the KPI-based network
    rect = patches.Rectangle(
        (0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', transform=axes[1].transAxes
    )
    axes[1].add_patch(rect)

    # Show the combined plot
    plt.tight_layout()
    plt.show()