import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



# SEED
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# METRICS


class MetricTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "Epoch": list(range(1, len(self.train_accs) + 1)),
                "Train Loss": self.train_losses,
                "Val Loss": self.val_losses,
                "Train Accuracy": self.train_accs,
                "Val Accuracy": self.val_accs,
            }
        )


# PLOTTING FUNCTIONS


def plot_loss_curve(tracker, suffix="model", output_dir="."):
    plt.figure(figsize=(10, 6))
    plt.plot(tracker.train_losses, label="Train Loss")
    plt.plot(tracker.val_losses, label="Val Loss")
    plt.title(f"Loss Curve - {suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"loss_{suffix}.png")
    plt.savefig(save_path, dpi=300)

    plt.close()


def plot_accuracy_curve(tracker, suffix="model", output_dir="."):
    plt.figure(figsize=(10, 6))
    plt.plot(tracker.train_accs, label="Train Accuracy")
    plt.plot(tracker.val_accs, label="Val Accuracy")
    plt.title(f"Accuracy Curve - {suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(output_dir, f"accuracy_{suffix}.png")
    plt.savefig(save_path, dpi=300)

    plt.close()


# CONFUSION MATRIX
def plot_confusion_matrix(model, dataloader, device, suffix="model", output_dir="."):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {suffix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(output_dir, f"cm_{suffix}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close()
    return cm


# NORMALIZED CONFUSION MATRIX


def plot_normalized_confusion_matrix(model, dataloader, device, output_dir="."):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, cmap="Blues")
    plt.title("Normalized Confusion Matrix")

    save_path = os.path.join(output_dir, "cm_normalized.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close()
    return cm_norm
