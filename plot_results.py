import matplotlib.pyplot as plt
import numpy as np

# Summary data extracted from logs (Manual Entry)
# Epochs: 1, 10, 20, 30, 40, 50, 59 (Best), 69 (End)
epochs = list(range(1, 70))

# Sample data points from training logs
train_loss = [5.96, 3.01, 2.15, 1.70, 1.41, 1.25, 1.12, 1.00] 
val_loss =   [5.08, 4.20, 4.04, 4.09, 4.13, 4.23, 4.20, 4.34]
f1_scores =  [0.35, 0.47, 0.48, 0.48, 0.50, 0.51, 0.52, 0.50]

# Interpolate data to match epoch length (for smoother visualization)
x_new = np.linspace(1, 69, 69)
x_old = np.linspace(1, 69, len(train_loss))

train_loss_interp = np.interp(x_new, x_old, train_loss)
val_loss_interp = np.interp(x_new, x_old, val_loss)
f1_interp = np.interp(x_new, x_old, f1_scores)

# --- Plot 1: Loss Chart ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss_interp, label='Train Loss (ResNet18)', color='blue')
plt.plot(epochs, val_loss_interp, label='Val Loss (ResNet18)', color='orange')

# Mark the best model epoch
plt.axvline(x=59, color='green', linestyle='--', label='Best Model (Epoch 59)')

plt.title('ResNet18 Baseline: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("resnet_loss_chart_fixed.png")
print("Loss chart saved: resnet_loss_chart_fixed.png")

# --- Plot 2: F1 Score Chart ---
plt.figure(figsize=(10, 5))
plt.plot(epochs, f1_interp, label='F1 Score (ResNet18)', color='green')

# Mark max F1 score
plt.axvline(x=59, color='red', linestyle='--', label='Max F1: 0.522')

plt.title('ResNet18 Baseline: F1 Score Progress')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.savefig("resnet_f1_chart_fixed.png")
print("F1 chart saved: resnet_f1_chart_fixed.png")