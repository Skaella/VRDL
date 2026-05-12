import os
from ultralytics import YOLO


def main():

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    yaml_path = os.path.join(SCRIPT_DIR, "medical.yaml")

    model = YOLO("yolov8m-seg.pt")

    # 2. Train the model
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=1024,  # High res for small cell detection
        batch=2,  # Low batch for memory management
        device=0,  # Ensure it uses the GPU
        patience=10,  # Stop early if validation mAP plateaus
        project=os.path.join(SCRIPT_DIR, "runs"),  # Save inside YOLO/runs
        name="yolo_medical_run",
        exist_ok=True,  # Overwrite/Resume in the same folder if it exists
        save=True,  # Explicitly save checkpoints
    )


if __name__ == "__main__":
    main()
