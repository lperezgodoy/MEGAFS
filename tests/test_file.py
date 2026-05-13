import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression # Not a classifier, used for testing invalid input
from sklearn.metrics import make_scorer, f1_score

from megafs import megafs

def test_file_not_found():
    with pytest.raises(FileNotFoundError, match="was not found"):
        validate_dataset("file_false.xlsx")

def test_validate_dataset_read_error(monkeypatch, tmp_path):
    fake_file = tmp_path / "test.xlsx"
    fake_file.write_text("dummy")

    
    def mock_read_excel(*args, **kwargs):
        raise RuntimeError("Simulated Pandas internal error")
    
    monkeypatch.setattr(pd, "read_excel", mock_read_excel)

    with pytest.raises(ValueError, match="Failed to parse Excel file"):
        validate_dataset(str(fake_file))

def test_empty_dataframe(tmp_path):
    path = tmp_path / "empty.xlsx"
    pd.DataFrame().to_excel(path, index=False)
    
    with pytest.raises(ValueError, match="The DataFrame is empty"):
        validate_dataset(str(path))

def test_missing_class_column(tmp_path):
    path = tmp_path / "no_class.xlsx"
    df = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    df.to_excel(path, index=False)
    
    with pytest.raises(KeyError, match="Required column 'class' is missing"):
        validate_dataset(str(path))

def test_non_numeric_columns(tmp_path):
    path = tmp_path / "strings.xlsx"
    df = pd.DataFrame({'class': [0, 1], 'data': ['a', 'b']})
    df.to_excel(path, index=False)
    
    with pytest.raises(TypeError, match="Non-numeric columns found"):
        validate_dataset(str(path))

def test_successful_loading(tmp_path):
    path = tmp_path / "valid.xlsx"
    df_input = pd.DataFrame({'val': [10.5, 20.0], 'class': [0, 1],})
    df_input.to_excel(path, index=False)
    
    df_output = validate_dataset(str(path))
    
    assert not df_output.empty
    assert 'class' in df_output.columns
    assert df_output.select_dtypes(exclude=['number']).empty

def test_validate_classifier_dict_success():
    modelos_validos = {
        "rf": RandomForestClassifier(),
        "rf_2": RandomForestClassifier()
    }
    assert validate_classifier_dict(modelos_validos) is True

def test_validate_classifier_dict_not_a_dict():
    with pytest.raises(TypeError, match="is not a dictionary"):
        validate_classifier_dict(["no", "soy", "un", "diccionario"])

def test_validate_classifier_dict_invalid_model():
    modelos_mixtos = {
        "clasificador": RandomForestClassifier(),
        "regresor": LinearRegression()  # Esto debería hacer fallar la función
    }
    with pytest.raises(TypeError, match="is not a valid scikit-learn classifier"):
        validate_classifier_dict(modelos_mixtos)

def test_valid_string_metric():
    """Tests if common string metrics like 'accuracy' are accepted."""
    assert validate_sklearn_classifier_metric("accuracy") is True
    assert validate_sklearn_classifier_metric("f1") is True

def test_valid_callable_metric():
    """Tests if a custom scorer object (made with make_scorer) is accepted."""
    custom_scorer = make_scorer(f1_score)
    assert validate_sklearn_classifier_metric(custom_scorer) is True

def test_invalid_string_metric():
    """Tests if a non-existent metric name raises a ValueError."""
    with pytest.raises(ValueError, match="is not a valid scikit-learn metric"):
        validate_sklearn_classifier_metric("not_a_real_metric")

def test_invalid_type_metric():
    """Tests if passing an integer or list raises a TypeError."""
    with pytest.raises(TypeError, match="must be a string or a callable"):
        validate_sklearn_classifier_metric(123)
