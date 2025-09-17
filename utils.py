from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

# Load model
MODEL_PATH = "model/plant_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Load recommendations
RECOMMENDATIONS_PATH = "recommendations/recommendations.json"
if not os.path.exists(RECOMMENDATIONS_PATH):
    raise FileNotFoundError(f"Recommendations file not found: {RECOMMENDATIONS_PATH}")
with open(RECOMMENDATIONS_PATH, "r") as f:
    recommendations = json.load(f)

# Class labels (same order as training / JSON keys)
class_labels = list(recommendations.keys())

# Disease type mapping (Category + Specific Disease Name)
disease_types = {
    'Pepper__bell___Bacterial_spot': "Bacterial – Bacterial Spot",
    'Pepper__bell___healthy': "Healthy – No Disease",
    'Potato___Early_blight': "Fungal – Early Blight",
    'Potato___Late_blight': "Fungal – Late Blight",
    'Potato___healthy': "Healthy – No Disease",
    'Tomato_Bacterial_spot': "Bacterial – Bacterial Spot",
    'Tomato_Early_blight': "Fungal – Early Blight",
    'Tomato_Late_blight': "Fungal – Late Blight",
    'Tomato_Leaf_Mold': "Fungal – Leaf Mold",
    'Tomato_Septoria_leaf_spot': "Fungal – Septoria Leaf Spot",
    'Tomato_Spider_mites_Two_spotted_spider_mite': "Mite – Spider Mites (Two-spotted)",
    'Tomato__Target_Spot': "Fungal – Target Spot",
    'Tomato__Tomato_YellowLeaf__Curl_Virus': "Viral – Tomato Yellow Leaf Curl Virus",
    'Tomato__Tomato_mosaic_virus': "Viral – Tomato Mosaic Virus",
    'Tomato_healthy': "Healthy – No Disease"
}

def predict_disease(img_path):
    """Return predicted disease, type, recommendation, and confidence."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    disease = class_labels[class_index]   # raw disease label
    disease_type = disease_types.get(disease, "fungal infected")
    recommendation = recommendations.get(disease, "No recommendation available.")

    return disease, disease_type, recommendation, confidence
