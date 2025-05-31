"""
Functions to transform dataframes of domains, tables, fields.
"""

import numpy as np
import pandas as pd
import networkx as nx

def build_common_fields_matrix_schema(df: pd.DataFrame) -> np.ndarray:
    """
    Build a matrix where each row and column represents a table_name, and the content
    of the matrix is the number of common fields between the tables.

    Args:
        df (pd.DataFrame): Input DataFrame with at least 'table_name' and 'field_name' columns.

    Returns:
        np.ndarray: A square matrix where element (i, j) is the number of common fields
                    between table i and table j.
    """
    # Ensure required columns are present
    if not set(["table_name", "field_name"]).issubset(df.columns):
        raise ValueError("The DataFrame must contain 'table_name' and 'field_name' columns.")

    # Group by table_name and collect fields for each table
    table_fields = df.groupby("table_name")["field_name"].apply(set)

    # Get the list of table names
    table_names = table_fields.index.tolist()

    # Create a mapping of table_name to domain_name
    table_to_domain = df.drop_duplicates(subset="table_name").set_index("table_name")["domain_name"].to_dict()


    # Initialize an empty square matrix
    n = len(table_names)
    matrix = np.zeros((n, n), dtype=int)

    # Populate the matrix with the number of common fields
    for i, table_i in enumerate(table_names):
        for j, table_j in enumerate(table_names):
            if i < j:  # Only compute for upper triangle (matrix is symmetric)
                common_fields = table_fields[table_i].intersection(table_fields[table_j])
                matrix[i, j] = len(common_fields)
                matrix[j, i] = matrix[i, j]  # Symmetric value

    # Convert the matrix to a networkx graph for visualization
    G = nx.from_numpy_array(matrix, create_using=nx.Graph())

    return matrix, table_names, table_to_domain, G

def build_common_fields_matrix_kpis(df: pd.DataFrame) -> tuple[np.ndarray, list, dict, dict]:
    """
    Build a matrix where each row and column represents a table_name, and the content
    of the matrix indicates whether two tables are connected by belonging to the same KPI.

    Args:
        df (pd.DataFrame): Input DataFrame with at least 'kpi_name', 'table_name', and 'domain_name' columns.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A square matrix where element (i, j) is 1 if the tables are connected by the same KPI, 0 otherwise.
            - list: A list of table names corresponding to the rows/columns of the matrix.
            - dict: A dictionary mapping each table_name to its domain_name.
            - dict: A dictionary mapping each table_name to the list of kpi_name values it belongs to.
    """
    # Ensure required columns are present
    if not set(["kpi_name", "table_name", "domain_name"]).issubset(df.columns):
        raise ValueError("The DataFrame must contain 'kpi_name', 'table_name', and 'domain_name' columns.")

    # Get the list of unique table names
    table_names = df["table_name"].unique().tolist()

    # Create a mapping of table_name to domain_name
    table_to_domain = df.drop_duplicates(subset="table_name").set_index("table_name")["domain_name"].to_dict()

    # Create a mapping of table_name to kpi_name (list of KPIs)
    table_to_kpi = df.groupby("table_name")["kpi_name"].apply(list).to_dict()

    # Initialize an empty square matrix
    n = len(table_names)
    matrix = np.zeros((n, n), dtype=int)

    # Group by kpi_name and connect tables that belong to the same KPI
    kpi_groups = df.groupby("kpi_name")["table_name"].apply(list)
    for tables in kpi_groups:
        for i, table_i in enumerate(tables):
            for j, table_j in enumerate(tables):
                if table_i != table_j:  # Avoid self-loops
                    idx_i = table_names.index(table_i)
                    idx_j = table_names.index(table_j)
                    matrix[idx_i, idx_j] = 1
                    matrix[idx_j, idx_i] = 1  # Symmetric value

    # Convert the matrix to a networkx graph for visualization
    G = nx.from_numpy_array(matrix, create_using=nx.Graph())

    return matrix, table_names, table_to_domain, table_to_kpi



