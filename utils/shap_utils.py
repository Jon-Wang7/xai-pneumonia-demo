import shap
import numpy as np

def generate_shap_values(model, data, sample_size=10):
    """
    Generate SHAP values using GradientExplainer.
    """
    data = np.array(data)
    if data.shape[0] < 2:
        data = np.vstack([data, data])

    sample_data = data[:sample_size]
    explainer = shap.GradientExplainer(model, sample_data)
    shap_values = explainer.shap_values(sample_data)
    return shap_values



def save_shap_explanation(image, shap_values, output_file):
    """
    Saves the SHAP explanation as an image.
    Args:
        image: The original input image.
        shap_values: The SHAP values for the image.
        output_file: The file path where the SHAP explanation will be saved.
    """
    # Visualization code
    import matplotlib.pyplot as plt
    shap.image_plot([shap_values], [image], show=False)
    plt.savefig(output_file)
    plt.close()