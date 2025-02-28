import json
import matplotlib.pyplot as plt

epoch_losses_file = 'results/epoch_losses.json'
roc_curve_file = 'results/roc_curve.json'

with open(epoch_losses_file, 'r') as f:
    epoch_losses = json.load(f)

with open(roc_curve_file, 'r') as f:
    roc_curve = json.load(f)

plt.figure(figsize=(10, 5))
plt.plot(epoch_losses["train"], label="Training Loss", color="blue", marker="o")
plt.plot(epoch_losses["val"], label="Validation Loss", color="orange", marker="o")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses over Epochs')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(roc_curve["FPR"], roc_curve["TPR"], label="ROC Curve", color="green", marker="x")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Random Classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
