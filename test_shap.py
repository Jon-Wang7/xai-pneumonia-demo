import numpy as np
import shap
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt


# 加载模型
model_path = "models/vgg16_model.h5"
model = load_model(model_path)


def preprocess_image(image_path):
    """
    预处理图片函数
    """
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(image, axis=0), image


def generate_shap_values(model, data, background_data=None):
    """
    生成 SHAP 值
    """
    if background_data is None:
        background_data = data  # 默认使用输入图片作为背景

    explainer = shap.GradientExplainer(model, background_data)
    shap_values, indexes = explainer.shap_values(data, ranked_outputs=1)
    return shap_values, indexes


def save_shap_explanation(image, shap_values, output_file):
    """
    保存 SHAP 可视化图像
    """
    # 确保 shap_values 格式为列表
    if not isinstance(shap_values, list):
        shap_values = [shap_values]

    # 绘制 SHAP 可视化
    plt.figure(figsize=(8, 8))
    shap.image_plot(shap_values, [image], show=False)  # 输入图片需为列表格式
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    # 输入图片路径
    image_path = "static/person46_virus_96.jpeg"

    # 预处理图片
    processed_image, original_image = preprocess_image(image_path)

    try:
        # 使用图片本身作为背景数据
        shap_values, indexes = generate_shap_values(model, processed_image, processed_image)
        print("SHAP Values Generated Successfully.")

        # 保存 SHAP 可视化
        output_path = "static/outputs/shap/sample_image_shap_optimized.png"
        save_shap_explanation(original_image, shap_values[0], output_path)
        print(f"SHAP visualization saved at {output_path}")

        # 显示保存的结果
        plt.imshow(plt.imread(output_path))
        plt.axis('off')  # 隐藏坐标轴
        plt.show()
    except Exception as e:
        print("Error in generating SHAP values:", str(e))