import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ======================================================
# 📂 Directory setup
# ======================================================
data_dir = 'dataset'
model_path = 'model/plant_disease_model.h5'
history_path = 'model/history.pkl'

# ======================================================
# 📊 Load training history (Accuracy visualization)
# ======================================================
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(8, 5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model/training_validation_accuracy.png")
    plt.show()
else:
    print("⚠️ History file not found! Skipping accuracy plot.")

# ======================================================
# 🌿 Load model and visualize performance (Confusion Matrix)
# ======================================================
if os.path.exists(model_path):
    print("\n✅ Loading saved model...")
    model = load_model(model_path)

    print("📦 Loading validation dataset...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    print("🔍 Making predictions...")
    y_true = val_data.classes
    y_probs = model.predict(val_data)
    y_pred = np.argmax(y_probs, axis=1)
    labels = list(val_data.class_indices.keys())

    print("📉 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix - Plant Disease Classification")
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png")
    plt.show()

    print("\n✅ Visualization complete! Files saved in 'model/' folder.")
else:
    print("❌ Model file not found! Please check 'model/plant_disease_model.h5'.")
