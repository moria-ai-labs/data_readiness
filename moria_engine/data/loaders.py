"""
Functions to load datasets.
"""

import pandas as pd
import json

def load_csv(file_path):
    """Load a CSV file into a Pandas DataFrame."""
    return pd.read_csv(file_path)


def read_json(file_path):
    """
    Read a JSON file from the given file path and return it as a JSON object.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: The JSON object (parsed from the file).
    """
    with open(file_path, 'r') as file:
        return json.load(file)

def preprocess_nested_json_schema(nested_json):
    """Flatten a nested JSON structure into a list of dictionaries."""
    rows = []
    for domain in nested_json:
        domain_name = domain["domain_name"]
        for table in domain["tables"]:
            table_name = table["table_name"]
            for field in table["fields"]:
                rows.append({
                    "domain_name": domain_name,
                    "table_name": table_name,
                    "field_name": field["field_name"]
                })
    return pd.DataFrame(rows)



def preprocess_nested_json_kpis(nested_json):
    """
    Flatten a nested JSON structure for KPIs into a list of dictionaries containing
    'kpi_name', 'domain_name', 'table_name', and 'field_name'.

    Args:
        nested_json (list): A list of dictionaries representing the JSON structure.

    Returns:
        pd.DataFrame: A DataFrame containing 'kpi_name', 'domain_name', 'table_name', and 'field_name'.
    """
    rows = []
    for kpi in nested_json:
        kpi_name = kpi.get("kpi_name", "")  # Extract the kpi_name
        if "data_required" in kpi and isinstance(kpi["data_required"], list):
            for data in kpi["data_required"]:
                rows.append({
                    "kpi_name": kpi_name,
                    "domain_name": data.get("domain_name", ""),
                    "table_name": data.get("table_name", ""),
                    "field_name": data.get("field_name", "")
                })
    return pd.DataFrame(rows)

def load_json_schema(file_path):
    """Load a JSON file into a Pandas DataFrame."""
    nested_json = read_json(file_path)
    if isinstance(nested_json, list) and all(isinstance(item, dict) for item in nested_json):
        return preprocess_nested_json_schema(nested_json)
    else:
        # If the JSON is not nested, load it directly into a DataFrame
        return pd.read_json(file_path)

def load_json_kpis(file_path):
    """Load a JSON file into a Pandas DataFrame."""
    nested_json = read_json(file_path)
    if isinstance(nested_json, list) and all(isinstance(item, dict) for item in nested_json):
        return preprocess_nested_json_kpis(nested_json)
    else:
        # If the JSON is not nested, load it directly into a DataFrame
        return pd.read_json(file_path)