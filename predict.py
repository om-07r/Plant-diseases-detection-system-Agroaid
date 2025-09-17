import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. List your class names here in the same order as used during model training
class_names = [
   'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
    # Add full list if more classes exist...
]

# 2. Load the model
model_path = os.path.join("model", "plant_disease_model.h5")
if not os.path.exists(model_path):
    print(f"Model not found at: {model_path}")
    sys.exit()

model = load_model(model_path)

# 3. Check image path from command-line
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Image not found at: {image_path}")
    sys.exit()

print(f"Using image: {image_path}")

# 4. Preprocess
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 5. Predict
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index] if predicted_index < len(class_names) else "Unknown Class"

# 6. Output
print(f" Predicted Class: {predicted_class} (Index: {predicted_index})")
print(f" Prediction Confidence: {prediction[0][predicted_index]:.4f}")
