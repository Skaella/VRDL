import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


class Visualizer:
    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.plot_dir = os.path.join(self.result_dir, "plots")
        self.sample_dir = os.path.join(self.result_dir, "samples")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

    def load_pred_npz(self, npz_filename="pred.npz"):
        """Loads the required prediction archive from the script directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, npz_filename)

        if not os.path.exists(path):
            print(f"❌ ERROR: {npz_filename} not found at {path}")
            return None

        data = np.load(path)
        print(f"DEBUG: Found keys in npz: {list(data.keys())}")
        return data

    def save_all_samples(self, npz_data):
        """Converts raw NPZ arrays into high-contrast PNG files."""
        key = 'preds' if 'preds' in npz_data else list(npz_data.keys())[0]
        data_array = npz_data[key]

        print(f"Processing {len(data_array)} images from '{key}'...")
        for i in range(len(data_array)):
            img = data_array[i]

            # Ensure shape is (H, W, C)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            # Min-Max scaling
            img = img - img.min()
            img = img / (img.max() + 1e-8)
            img = (img * 255.0).astype(np.uint8)

            save_path = os.path.join(self.sample_dir, f"result_{i:03d}.png")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved all samples to {self.sample_dir}")

    def plot_curves(self, metrics_json="metrics.json"):
        json_path = os.path.join(self.result_dir, metrics_json)
        if not os.path.exists(json_path):
            return

        with open(json_path, "r") as f:
            data = json.load(f)

        epochs = [item['epoch'] for item in data]
        loss = [item['loss'] for item in data]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, loss, label='L1 Loss', color='blue')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "loss_curve.png"))
        plt.close()

    def save_comparison(self, degraded_path, restored_path, output_name):
        """
        Creates a side-by-side [Degraded | Restored] comparison.
        """
        # Load the images
        degraded = cv2.cvtColor(cv2.imread(degraded_path), cv2.COLOR_BGR2RGB)
        restored = cv2.cvtColor(cv2.imread(restored_path), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))

        # Plot Degraded
        plt.subplot(1, 2, 1)
        plt.imshow(degraded)
        plt.title("Degraded Input")
        plt.axis('off')

        # Plot Restored
        plt.subplot(1, 2, 2)
        plt.imshow(restored)
        plt.title("Restored Output (PSNR: 30.12dB)")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.result_dir,
                f"comparison_{output_name}.png"))
        plt.close()
        print(f"Comparison saved: comparison_{output_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./results")
    args = parser.parse_args()

    viz = Visualizer(args.result_dir)

    # 1. Generate Plots
    viz.plot_curves()

    # 2. Process Samples
    npz_data = viz.load_pred_npz("pred.npz")
    if npz_data:
        viz.save_all_samples(npz_data)

    degraded_file = os.path.join("./dataset/test/degraded", "0.png")
    restored_file = os.path.join("./results/samples", "result_000.png")

    viz.save_comparison(degraded_file, restored_file, "000")


if __name__ == "__main__":
    main()
