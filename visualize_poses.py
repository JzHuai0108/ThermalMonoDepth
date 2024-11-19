import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D viz
import argparse


def cat_poses(poses):
    """

    Args:
        poses: [N, 5, 3, 4]. 5 is subsequence length, N is number of subsequences, 3x4 is [R, t] of C(j)_T_C(j+i).
    Returns:
        traj: [N+4, 4, 4] W_T_Ci poses
    """
    N, subseq = poses.shape[:2]
    imgs = N + subseq - 1
    traj = []

    # Initialize the trajectory with the identity matrix for C(0)_T_C(0)
    traj.append(np.eye(4))

    # Build the trajectory
    for j in range(poses.shape[0]):
        if j == N - 1:
            # Handle the last subsequence with the full subsequence length
            C0_T_Cj = traj[-1]
            for i in range(1, subseq):  # Iterate from 1 to (subseq - 1)
                Cj_T_Cjpi = poses[j, i, :, :]  # Get the pose matrix
                # Append the last row [0, 0, 0, 1] to make it 4x4
                Cj_T_Cjpi = np.vstack((Cj_T_Cjpi, np.array([0, 0, 0, 1])))
                traj.append(C0_T_Cj @ Cj_T_Cjpi)
        else:
            # Handle the intermediate subsequences
            Cj_T_Cjp1 = poses[j, 1, :, :]  # Get the pose matrix for the first step in the subsequence
            # Append the last row [0, 0, 0, 1] to make it 4x4
            Cj_T_Cjp1 = np.vstack((Cj_T_Cjp1, np.array([0, 0, 0, 1])))
            traj.append(traj[-1] @ Cj_T_Cjp1)

    # Assert that the trajectory length matches the number of images
    assert len(traj) == imgs
    return traj


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
        positions.append(pose[:3, 3])  # Get the translation vector (x, y, z) from each 4x4 pose matrix

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
