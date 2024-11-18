import numpy as np
import matplotlib.pyplot as plt
import time

# Load the .npy file containing multiple depth images
import numpy as np
import matplotlib.pyplot as plt
import time

# Load the .npy file containing multiple depth images
npy_file_path = '/home/pi/Desktop/temp/results/Indoor_model/Depth/gym/predictions.npy'
depth_images = np.load(npy_file_path)  # Assuming shape (num_images, height, width)


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
def visualize_depth_images(depth_images, delay=1):
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()

    for i, depth_image in enumerate(depth_images):
        ax.clear()
        ax.imshow(depth_image, cmap='viridis', vmin=0, vmax=255)  # Display the normalized depth image
        ax.set_title(f'Depth Image {i + 1}/{len(depth_images)}')
        plt.pause(delay)  # Pause for 'delay' seconds between images

    plt.ioff()  # Disable interactive mode
    plt.show()


# Normalize and visualize the depth images
depth_images_normalized = normalize_depth_images(depth_images)
visualize_depth_images(depth_images_normalized, delay=1)