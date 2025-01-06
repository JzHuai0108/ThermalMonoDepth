import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D viz
import argparse
from common.utils.convert_poses import cat_poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize poses estimated by ThermalMonoDepth")
    parser.add_argument('--model', type=str, default='T_vivid_resnet18_indoor', help='Model name')
    parser.add_argument('--seq', type=str, default='indoor_robust_varying', help='Sequence name')
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

    fn = os.path.join(resultdir, model, 'POSE', seq, 'predictions.npy')
    gt = os.path.join(datadir, seq, 'poses_T.txt')
    poses = np.load(fn)
    traj = cat_poses(poses)

    # Extract positions from the trajectory matrices
    positions = []
    for pose in traj:
        positions.append(pose[:3])  # Get the translation vector (x, y, z) from each 4x4 pose matrix

    positions = np.array(positions) # Convert to a NumPy array for further use

    gt_poses = np.loadtxt(gt)
    gt_poses = np.reshape(gt_poses, (-1, 3, 4))
    gt_positions = gt_poses[:, :, 3]

    num_gt_poses = gt_positions.shape[0]
    start = 0
    end = num_gt_poses // 2
    scale_factor = np.sum(gt_positions[start:end, :] * positions[start:end, :])/np.sum(positions[start:end, :] ** 2)
    print(f'scale_factor {scale_factor}')
    positions *= scale_factor

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r--', label='ThermalMonoDepth')
    ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], 'g-', label='Ref')
    ax.plot(gt_positions[0, 0], gt_positions[0, 1], gt_positions[0, 2], 'ko', label='Start')
    ax.plot(gt_positions[-1, 0], gt_positions[-1, 1], gt_positions[-1, 2], 'bs', label='End')

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory')
    ax.legend()
    plt.show()
