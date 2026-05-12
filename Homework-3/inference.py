import os
import json
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from pycocotools import mask as mask_util
from tqdm import tqdm
import argparse


def main():
    # path setup
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    DEFAULT_WEIGHTS = os.path.join(
        SCRIPT_DIR, "runs", "yolo_medical_run", "weights", "best.pt"
    )
    DEFAULT_TEST_DIR = os.path.join(SCRIPT_DIR, "hw3-data-release", "test_release")
    DEFAULT_MAPPING = os.path.join(
        SCRIPT_DIR, "hw3-data-release", "test_image_name_to_ids.json"
    )
    DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "test-results.json")

    # 1. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--test_dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--mapping", type=str, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Safety Check
    if not os.path.exists(args.weights):
        print(f"❌ ERROR: Weights not found at {args.weights}")
        print("Wait for training to finish its first few epochs!")
        return

    # 2. Load Mapping
    with open(args.mapping, "r") as f:
        mapping_data = json.load(f)
    name_to_id = {item["file_name"]: item["id"] for item in mapping_data}

    # 3. Load Model
    print(f"Loading YOLO model from:\n{args.weights}")
    model = YOLO(args.weights)

    # 4. Process Test Images
    if not os.path.exists(args.test_dir):
        print(f"❌ ERROR: Test directory not found at {args.test_dir}")
        return

    test_images = [f for f in os.listdir(args.test_dir) if f.endswith(".tif")]
    print(f"Processing {len(test_images)} test images...")

    submission_results = []

    for img_name in tqdm(test_images):
        img_id = name_to_id.get(img_name)
        if img_id is None:
            continue

        img_path = os.path.join(args.test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Inference with 0.05 confidence for maximum recall
        results = model(img, imgsz=1024, conf=0.05, verbose=False)[0]

        if results.masks is None:
            continue

        # Extract predictions
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        masks = results.masks.data.cpu().numpy()

        for i in range(len(scores)):
            mask = masks[i]
            if mask.shape != (orig_h, orig_w):
                mask = cv2.resize(
                    mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

            binary_mask = (mask > 0.5).astype(np.uint8)
            if binary_mask.sum() == 0:
                continue

            # COCO RLE Encoding
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle["counts"] = rle["counts"].decode("utf-8")

            # COCO Bbox [x, y, width, height] with clipping safety
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
            x2, y2 = min(float(orig_w), float(x2)), min(float(orig_h), float(y2))
            bbox = [x1, y1, x2 - x1, y2 - y1]

            submission_results.append(
                {
                    "image_id": int(img_id),
                    "bbox": bbox,
                    "score": float(scores[i]),
                    "category_id": int(classes[i] + 1),
                    "segmentation": {
                        "size": [int(orig_h), int(orig_w)],
                        "counts": rle["counts"],
                    },
                }
            )

    # 5. Save the file
    with open(args.output, "w") as f:
        json.dump(submission_results, f)

    print(f"\nDone! Saved {len(submission_results)} predictions to:\n{args.output}")


if __name__ == "__main__":
    main()
