import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from dataloader import get_dataloaders
from model import PromptIR


def run_inference():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_WEIGHTS = os.path.join(
        SCRIPT_DIR, "checkpoints", "promptir_best.pth")
    DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "pred.npz")

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"❌ ERROR: Missing weights at {args.weights}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = PromptIR(dim=48, num_blocks=[4, 6, 6, 8], decoder=True).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load Test Dataloader
    _, _, test_loader = get_dataloaders(
        os.path.join(SCRIPT_DIR, "dataset"), batch_size=1)

    submission_dict = {}
    print("Running competition inference...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            degraded = batch['degraded'].to(device)
            full_path = batch['degraded_path'][0]
            file_name = os.path.basename(full_path)

            output = model(degraded)

            # Post-process: Convert normalized tensor to (3, H, W)
            img_np = output.squeeze(0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_np = (img_np * std) + mean

            img_np = img_np.transpose(1, 2, 0)
            img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
            img_np = img_np.transpose(2, 0, 1)
            submission_dict[file_name] = img_np

    np.savez(args.output, **submission_dict)
    print(f"Prediction saved to {args.output}")


if __name__ == "__main__":
    run_inference()
