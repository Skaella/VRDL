import os
import cv2
import argparse
import numpy as np
import tifffile
from tqdm import tqdm

def mask_to_yolo_polygons(mask_array, class_id, img_w, img_h):
    polygons = []
    instance_ids = np.unique(mask_array)
    instance_ids = instance_ids[instance_ids != 0]

    for obj_id in instance_ids:
        binary_mask = (mask_array == obj_id).astype(np.uint8)
        if np.sum(binary_mask) <= 3: 
            continue

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 3:
                contour = contour.flatten()
                normalized = []
                for i in range(len(contour)):
                    if i % 2 == 0: # X
                        normalized.append(contour[i] / img_w)
                    else:          # Y
                        normalized.append(contour[i] / img_h)
                
                # Format: "class_id x1 y1 x2 y2 ..."
                poly_str = f"{class_id - 1} " + " ".join([f"{val:.5f}" for val in normalized])
                polygons.append(poly_str)
    return polygons

def main():
    # --- UNIVERSAL PATH SETUP ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Defaults relative to the script location
    DEFAULT_INPUT = os.path.join(SCRIPT_DIR, 'hw3-data-release', 'train')
    DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'yolo_dataset')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    root_dir = args.input_dir
    out_dir = args.output_dir
    
    # Create YOLO structure (Train and Val)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    
    if not os.path.exists(root_dir):
        print(f"❌ Error: Input directory not found at {root_dir}")
        return

    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Sort subdirs to ensure consistent splitting
    subdirs.sort()
    
    # Use 10% for validation to monitor mAP during training
    val_size = max(1, int(len(subdirs) * 0.1))
    train_folders = subdirs[val_size:]
    val_folders = subdirs[:val_size]

    print(f"Found {len(subdirs)} samples. Splitting: {len(train_folders)} train, {len(val_folders)} val.")

    for split, folders in [("train", train_folders), ("val", val_folders)]:
        print(f"Processing {split} split...")
        for folder_name in tqdm(folders):
            img_dir = os.path.join(root_dir, folder_name)
            img_path = os.path.join(img_dir, "image.tif")
            
            if not os.path.exists(img_path): continue
                
            # Read and Save Image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
            img_h, img_w = img.shape[:2]
            cv2.imwrite(os.path.join(out_dir, f"images/{split}/{folder_name}.jpg"), img)
            
            # Process Masks
            all_polygons = []
            for class_id in range(1, 5):
                mask_path = os.path.join(img_dir, f"class{class_id}.tif")
                if os.path.exists(mask_path):
                    mask_array = tifffile.imread(mask_path)
                    polys = mask_to_yolo_polygons(mask_array, class_id, img_w, img_h)
                    all_polygons.extend(polys)
                    
            # Save Labels
            label_path = os.path.join(out_dir, f"labels/{split}/{folder_name}.txt")
            with open(label_path, "w") as f:
                f.write("\n".join(all_polygons))

    print(f"\n✅ Conversion complete! Data saved to {out_dir}")

if __name__ == "__main__":
    main()
