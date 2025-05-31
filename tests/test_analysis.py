import pytest
import pandas as pd
import os
from moria_engine.data.loaders import load_csv, load_json_schema 


def test_build_common_fields_matrix_kpis():
    """
    Test the build_common_fields_matrix_kpis function.
    Input: A sample DataFrame with KPI data.
    Validate: The output matrix and mappings are correct.
    """

    df = pd.DataFrame({
        "kpi_name": ["KPI1", "KPI1", "KPI2"],
        "domain_name": ["Domain1", "Domain1", "Domain2"],
        "table_name": ["Table1", "Table2", "Table3"]
    })
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_kpis
    matrix, table_names, table_to_domain, table_to_kpi = build_common_fields_matrix_kpis(df)

    assert matrix.shape == (3, 3)
    assert table_to_kpi["Table1"] == ["KPI1"]


def test_build_common_fields_matrix_schema():
    """
    Test the build_common_fields_matrix_schema function.
    Input: A sample DataFrame with schema data.
    Validate: The output matrix and mappings are correct.
    """


    df = pd.DataFrame({
        "domain_name": ["Domain1", "Domain1"],
        "table_name": ["Table1", "Table2"],
        "field_name": ["Field1", "Field1"]
    })
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema
    matrix, table_names, table_to_domain, G = build_common_fields_matrix_schema(df)

    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == 1  # One common field


def test_visualize_combined_networks1():
    """
    Test the visualize_combined_networks1 function.
    Input: Sample DataFrames for schema and KPI data.
    Validate: The function runs without errors.
    """


    df_schema = pd.DataFrame({
        "domain_name": ["Domain1", "Domain1"],
        "table_name": ["Table1", "Table2"],
        "field_name": ["Field1", "Field2"]
    })
    df_kpis = pd.DataFrame({
        "kpi_name": ["KPI1"],
        "domain_name": ["Domain1"],
        "table_name": ["Table1"]
    })
    from moria_engine.analysis.graph_visualizer import visualize_combined_networks1
    visualize_combined_networks1(df_schema, df_kpis)




