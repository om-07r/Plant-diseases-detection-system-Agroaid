import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Directory
data_dir = 'dataset'

# Reload validation data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load model
model = load_model("model/plant_disease_model.h5")

# Predict
y_true = val_data.classes
y_probs = model.predict(val_data)
y_pred = np.argmax(y_probs, axis=1)
labels = list(val_data.class_indices.keys())

# 📉 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show() 