from tensorflow.keras.models import load_model
import pickle

def load_cnn_model(path):
    """
    Load a CNN model from the given path.
    """
    return load_model(path)

def load_random_forest_model(path):
    """
    Load a Random Forest model from the given path.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

def load_models(cnn_path, rf_path):
    """
    Load both CNN and Random Forest models.
    """
    cnn_model = load_cnn_model(cnn_path)
    rf_model = load_random_forest_model(rf_path)
    return cnn_model, rf_model