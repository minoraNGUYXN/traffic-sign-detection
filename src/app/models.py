from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import cv2
import os

# Lấy đường dẫn gốc của project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn mô hình / Model paths
DETECTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolo11n.pt")
RECOGNITION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "traffic_sign_vgg16.h5")

# Danh sách nhãn biển báo
TRAFFIC_SIGN_CLASSES = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}


def load_models():
    """
    Tải các mô hình YOLO và Classification
    Load YOLO and Classification models

    Returns:
        tuple: (detection_model, classification_model)
    """
    try:
        print(f"Loading YOLO model from: {DETECTION_MODEL_PATH}")
        print(f"Loading classification model from: {RECOGNITION_MODEL_PATH}")

        # Kiểm tra file tồn tại
        if not os.path.exists(DETECTION_MODEL_PATH):
            print(f"YOLO model file not found: {DETECTION_MODEL_PATH}")
            return None, None

        if not os.path.exists(RECOGNITION_MODEL_PATH):
            print(f"Classification model file not found: {RECOGNITION_MODEL_PATH}")
            return None, None

        detection_model = YOLO(DETECTION_MODEL_PATH)
        classification_model = tf.keras.models.load_model(RECOGNITION_MODEL_PATH)

        print("Models loaded successfully!")
        return detection_model, classification_model

    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None


def preprocess_for_classification(image_crop):
    """
    Tiền xử lý ảnh biển báo để phân loại
    Preprocess traffic sign crop for classification

    Args:
        image_crop (np.ndarray): Ảnh biển báo đã cắt

    Returns:
        np.ndarray: Ảnh đã được tiền xử lý
    """
    # Resize to 32x32 (model input size)
    resized = cv2.resize(image_crop, (32, 32))

    # Convert to float and normalize
    normalized = resized.astype("float32") / 255.0

    # Add batch dimension
    preprocessed = np.expand_dims(normalized, axis=0)

    return preprocessed


def get_traffic_sign_name(class_id):
    """
    Lấy tên biển báo từ class ID
    Get traffic sign name from class ID

    Args:
        class_id (int): ID của lớp biển báo

    Returns:
        str: Tên biển báo
    """
    return TRAFFIC_SIGN_CLASSES.get(class_id, "Unknown")