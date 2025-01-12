import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from utils.gradcam_utils import generate_gradcam, save_gradcam
from utils.shap_utils import save_shap_explanation, generate_shap_values
from utils.preprocess import preprocess_image
import shap

# 创建必要的文件夹
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs/gradcam", exist_ok=True)
os.makedirs("static/outputs/shap", exist_ok=True)

# 加载模型
cnn_model = load_model("models/baseline_cnn_model.h5")
cnn_model.compile()  # 确保加载后编译
vgg16_model = load_model("models/vgg16_model.h5")
vgg16_model.compile()
with open("models/random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

# 创建 FastAPI 应用
app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 根路径重定向到前端页面
@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

# 转换预测结果为分类标签
def get_class_label(prediction, threshold=0.5):
    """
    Convert model prediction to class label.
    Args:
        prediction: Model output (probability or binary).
        threshold: Threshold for binary classification.
    Returns:
        "pneumonia" or "normal"
    """
    if isinstance(prediction, (float, np.floating)):
        return "pneumonia" if prediction >= threshold else "normal"
    elif isinstance(prediction, (int, np.integer)):
        return "pneumonia" if prediction == 1 else "normal"
    else:
        raise ValueError(f"Unsupported prediction type: {type(prediction)}")

# 预测接口
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # 保存上传文件
        file_path = f"static/uploads/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # 加载并预处理图片
        processed_image = preprocess_image(file_path)

        # 确保输入形状为 (batch_size, 150, 150, 3)
        processed_image = np.expand_dims(processed_image[0], axis=0)

        # 模型预测
        baseline_pred = cnn_model.predict(processed_image)[0][0]
        vgg16_pred = vgg16_model.predict(processed_image)[0][0]
        rf_pred = random_forest_model.predict(processed_image.flatten().reshape(1, -1))[0]

        # 打印调试信息
        print(f"Baseline CNN Prediction: {baseline_pred}")
        print(f"VGG16 Prediction: {vgg16_pred}")
        print(f"Random Forest Prediction: {rf_pred}")

        # Grad-CAM 可视化
        gradcam_file = f"static/outputs/gradcam/{file.filename}_gradcam.png"
        gradcam_heatmap = generate_gradcam(vgg16_model, processed_image[0], "block5_conv3")
        save_gradcam(processed_image[0], gradcam_heatmap, gradcam_file)

        print(f"Processed Image Type: {type(processed_image)}")
        print(f"Processed Image Shape: {np.array(processed_image).shape}")

        # Ensure processed_image is correctly formatted
        # processed_image_array = np.array(processed_image)
        #
        # print(f"SHAP Input Type: {type(processed_image_array)}")
        # print(f"SHAP Input Shape: {processed_image_array.shape}")

        # # Generate SHAP values
        # shap_values = generate_shap_values(vgg16_model, processed_image_array)
        # shap_file = f"static/outputs/shap/{file.filename}_shap.png"
        # save_shap_explanation(processed_image[0], shap_values[0], shap_file)

        # 转化为分类标签
        baseline_label = get_class_label(baseline_pred)
        vgg16_label = get_class_label(vgg16_pred)
        rf_label = get_class_label(rf_pred)

        # '/Users/jon/Codes/MyProjects/PythonProjects/xai-pneumonia-demo/' +

        # 返回预测结果
        return {
            "Baseline CNN Prediction": baseline_label,
            "VGG16 Prediction": vgg16_label,
            "Random Forest Prediction": rf_label,
            "Grad-CAM": gradcam_file,
            # "SHAP": shap_file,
        }
    except Exception as e:
        return {"error": str(e)}

# 图像预处理函数
def preprocess_image(file_path):
    """
    Preprocess the uploaded image for model input.
    Args:
        file_path: Path to the uploaded image.
    Returns:
        Preprocessed image with shape (1, 150, 150, 3).
    """
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    image = load_img(file_path, target_size=(150, 150))
    image = img_to_array(image) / 255.0
    return np.expand_dims(image, axis=0)