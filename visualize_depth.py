import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

# Normalize depth images to range [0, 255]
def normalize_depth_images(depth_images):
    # Normalize each image independently
    depth_images_normalized = []
    for depth_image in depth_images:
        min_val = np.min(depth_image)
        max_val = np.max(depth_image)
        # Normalize to [0, 255]
        normalized = (depth_image - min_val) / (max_val - min_val) * 255.0
        depth_images_normalized.append(normalized)
    return np.array(depth_images_normalized)


# Visualize the depth images sequentially
def visualize_depth_images(depth_images, delay_secs=0.1):
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()

    for i, depth_image in enumerate(depth_images):
        ax.clear()
        im = ax.imshow(depth_image, cmap='viridis', vmin=0, vmax=255)
        # Add a color bar for the depth image
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Depth Scale')  # Optional: label for the color bar

        ax.set_title(f'Depth Image {i + 1}/{len(depth_images)}')
        plt.pause(delay_secs)  # Pause for 'delay_secs' seconds between images
        # Remove the color bar after each iteration to prevent overlapping in the next loop
        cbar.remove()

    plt.ioff()  # Disable interactive mode
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize depth images estimated by ThermalMonoDepth")
    parser.add_argument('--model', type=str, default='T_vivid_resnet18_indoor', help='Model name')
    parser.add_argument('--seq', type=str, default='indoor_robust_dark', help='Sequence name')
    parser.add_argument('--datadir', type=str, default='datasets/VIVID_256', help='Path to the dataset directory')
    parser.add_argument('--resultdir', type=str, default='results', help='Path to the results directory')
    args = parser.parse_args()
    datadir = args.datadir
    resultdir = args.resultdir
    model = args.model
    seq = args.seq
    assert ('indoor' in model and 'indoor' in seq) or ('outdoor' in model and 'outdoor' in seq), \
        "The model and sequence must be consistently 'indoor' or 'outdoor'."

    # Print parsed arguments for verification (optional)
    print(f"Data Directory: {datadir}")
    print(f"Result Directory: {resultdir}")
    print(f"Model: {model}")
    print(f"Sequence: {seq}")

    fn = os.path.join(resultdir, model, 'Depth', seq, 'predictions.npy')
    depth_images = np.load(fn)  # Assuming shape (num_images, height, width)

    # Normalize and visualize the depth images
    depth_images_normalized = normalize_depth_images(depth_images)
    visualize_depth_images(depth_images_normalized, delay_secs=0.1)
