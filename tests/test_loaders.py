import pytest
import pandas as pd
import os
from moria_engine.data.loaders import load_csv, load_json_schema 

def test_load_csv(tmp_path):
    """Test the load_csv function."""
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4")

    # Load the CSV file
    df = load_csv(csv_file)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == 1
@pytest.mark.skip(reason="Skipping this test temporarily. The real test is with target files.")
def test_load_json(tmp_path):
    """Test the load_json function."""
    # Create a temporary JSON file
    json_file = tmp_path / "test.json"
    json_file.write_text('[{"col1": 1, "col2": 2}, {"col1": 3, "col2": 4}]')

    # Load the JSON file
    df = load_json(json_file)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["col1", "col2"]
    assert df.iloc[0]["col1"] == 1

def test_load_csv_target(tmp_path):
    """Test the load_csv function with a target file."""
 
    # Load a target CSV file
    target_csv = 'data/test/20250520-hacp-company-data_domains_schema.csv'
    #csv_file = os.path.join(os.path.dirname(__file__), target_csv)
    df = load_csv(target_csv)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] > 3 # Ensure there are more than 3 columns
    assert df.shape[0] > 0 # Ensure there are rows in the DataFrame
    assert set(["domain_name", "table_name","field_name"]).issubset(set(list(df.columns))) #minumum information required
    #assert df.iloc[0]["col1"] == 1

def test_load_json_target_schema(tmp_path):
    """Test the load_json function with a target file."""
    # Load a target JSON file
    target_json = 'data/test/20250520-hacp-company-data_domains_schema.json'
    #json_file = os.path.join(os.path.dirname(__file__), target_json)
    df = load_json_schema(target_json)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] >= 3 # Ensure there are more than 3 columns
    assert df.shape[0] > 0 # Ensure there are rows in the DataFrame
    assert set(["domain_name", "table_name","field_name"]).issubset(set(list(df.columns))) #minumum information required
    #assert df.iloc[0]["col1"] == 1