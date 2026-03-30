import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import json
import os
from tqdm import tqdm
from PIL import Image
from utils import (
    MetricTracker,
    plot_loss_curve,
    plot_accuracy_curve,
    plot_confusion_matrix,
)

# Make training faster
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NumericImageFolder(Dataset):
    """ImageFolder that sorts classes numerically instead of alphabetically"""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
            key=lambda x: int(x) if x.isdigit() else 999999,  # numeric sort
        )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        print(f"Class order (first 10): {self.classes[:10]}")
        print(f"Class order (last 10): {self.classes[-10:]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    # DIRECTORY SETUP
    FOLDER_NAME = "seed0_results"
    OUTPUT_DIR = os.path.join("outputs", FOLDER_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    LOG_PATH = os.path.join(OUTPUT_DIR, "logs_seed0.csv")
    MODEL_PATH = os.path.join(OUTPUT_DIR, "resnet50_seed0_best.pth")
    CLASS_MAP_PATH = os.path.join(OUTPUT_DIR, "class_to_idx.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    # CONFIGURATION
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")

    print(f"Using device: {device}")

    # DATA TRANSFORMS
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # LOAD DATASETS
    print("\nLoading datasets...")
    train_dataset = NumericImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = NumericImageFolder(VAL_DIR, transform=val_transform)

    # Save class mapping
    class_to_idx = train_dataset.class_to_idx
    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(class_to_idx, f)

    # Create correct mapping: model index -> actual label
    idx_to_label = {idx: int(class_name) for class_name, idx in class_to_idx.items()}

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # MODEL
    print("\nCreating model...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)

    # LOSS AND OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # TRAINING LOOP
    best_val_acc = 0.0
    tracker = MetricTracker()

   
    print("STARTING TRAINING")
   

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # TRAIN
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # VALIDATION
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        print(f"\nTrain Accuracy: {train_acc:.2f}%")
        print(f"Val Accuracy: {val_acc:.2f}%")

        tracker.update(
            train_loss / len(train_loader),
            val_loss / len(val_loader),
            train_acc,
            val_acc,
        )
        scheduler.step()

        df = tracker.to_dataframe()
        df.to_csv(LOG_PATH, index=False)

        # SAVE BEST MODEL
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_to_idx": class_to_idx,
                    "idx_to_label": idx_to_label,
                },
                MODEL_PATH,
            )
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")

   
    plot_loss_curve(tracker, suffix="seed0", output_dir=OUTPUT_DIR)
    plot_accuracy_curve(tracker, suffix="seed0", output_dir=OUTPUT_DIR)
    plot_confusion_matrix(
        model, val_loader, device, suffix="seed0", output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
