import pickle
import matplotlib.pyplot as plt

# Load history
with open("model/history.pkl", "rb") as f:
    history = pickle.load(f)

# Plot
plt.plot(history['accuracy'], label='train')
plt.plot(history['val_accuracy'], label='val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("model/versus.png")
plt.show()
