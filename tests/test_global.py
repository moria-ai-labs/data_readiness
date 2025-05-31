import pytest
import pandas as pd
import os
from moria_engine.data.loaders import load_csv, load_json_schema 


def test_end_to_end_workflow():
    """ 
    Test the end-to-end workflow of loading data, transforming it, and visualizing it. 
    1. Integration Tests
    Test End-to-End Workflow:
    Input: A sample dataset for schema and KPI analysis.
    Validate: The outputs of schema-based and KPI-based matrices, and ensure the visualizations are generated without errors.
    """
    
    # Sample schema data
    df_schema = pd.DataFrame({
        "domain_name": ["Domain1", "Domain1", "Domain2"],
        "table_name": ["Table1", "Table2", "Table3"],
        "field_name": ["Field1", "Field2", "Field3"]
    })

    # Sample KPI data
    df_kpis = pd.DataFrame({
        "kpi_name": ["KPI1", "KPI2"],
        "domain_name": ["Domain1", "Domain2"],
        "table_name": ["Table1", "Table2"]
    })

    # Generate matrices
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema, build_common_fields_matrix_kpis
    schema_matrix, _, _, _ = build_common_fields_matrix_schema(df_schema)
    kpi_matrix, _, _, _ = build_common_fields_matrix_kpis(df_kpis)

    # Validate matrices are not empty
    assert schema_matrix.shape[0] > 0
    assert kpi_matrix.shape[0] > 0

    # Test visualization
    from moria_engine.analysis.graph_visualizer import visualize_combined_networks1
    visualize_combined_networks1(df_schema, df_kpis)

def test_large_dataset_performance():
    """
    Test the performance of the data transformation functions with a large dataset.
    Input: A large dataset for schema analysis.
    Validate: The function completes within a reasonable time frame.
    """


    import time
    # Generate a large dataset
    df_schema = pd.DataFrame({
        "domain_name": ["Domain" + str(i % 10) for i in range(1000)],
        "table_name": ["Table" + str(i) for i in range(1000)],
        "field_name": ["Field" + str(i % 100) for i in range(1000)]
    })

    start_time = time.time()
    from moria_engine.analysis.data_transformers import build_common_fields_matrix_schema
    build_common_fields_matrix_schema(df_schema)
    elapsed_time = time.time() - start_time

    # Ensure the process completes within a reasonable time
    assert elapsed_time < 5  # Example threshold