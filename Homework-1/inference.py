import os
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    TEST_DIR = os.path.join(BASE_DIR, "data", "test")

    MODEL_PATHS = [
        os.path.join(BASE_DIR, "test_outputs", "seed0_results", "resnet50_seed0_best.pth"),
        os.path.join(BASE_DIR, "test_outputs", "seed42_results", "resnet50_seed42_best.pth"),
        os.path.join(BASE_DIR, "test_outputs", "sgd_results", "resnet50_sgd_best.pth"),
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # LOAD MODELS

    models_list = []
    idx_to_label = None

    for path in MODEL_PATHS:
        if not os.path.exists(path):
            print(f"Warning: Model file not found at {path}. Skipping.")
            continue

        checkpoint = torch.load(path, map_location=device)

        if idx_to_label is None:
            idx_to_label = checkpoint["idx_to_label"]

        model = models.resnet50(weights=None)
        num_classes = len(checkpoint["class_to_idx"])
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        models_list.append(model)

    if not models_list:
        print("Error: No models were loaded. Please check your model paths.")
        return

    print(f"Loaded {len(models_list)} models")

    # TRANSFORM

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(TEST_DIR):
        print(f"Error: Test directory not found at {TEST_DIR}")
        return

    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith((".jpg"))])

    print(f"Found {len(test_images)} test images")

    # INFERENCE

    predictions = []

    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Ensemble Inference"):

            img_path = os.path.join(TEST_DIR, img_name)
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0).to(device)

            logits_sum = None

            for model in models_list:
                outputs = model(img_tensor)

                if logits_sum is None:
                    logits_sum = outputs
                else:
                    logits_sum += outputs

            # Final ensemble output
            outputs = logits_sum / len(models_list)

            pred_idx = torch.argmax(outputs, dim=1).item()
            pred_label = idx_to_label[pred_idx]

            img_id = os.path.splitext(img_name)[0]
            predictions.append([img_id, pred_label])

    # SAVE CSV

    output_file = "prediction.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        writer.writerows(predictions)

    print(f"\nSaved {output_file}")
    print(f"Total predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
