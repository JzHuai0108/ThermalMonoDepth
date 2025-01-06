import numpy as np
from scipy.spatial.transform import Rotation


def Pq_from_T(pose):
    """
    Convert a 4x4 transform matrix to pq [px py pz qx qy qz qw].
    pose is a 4x4 numpy array for SE3 transformation.

    Args:
        pose: 4x4 numpy array representing an SE3 transformation.
    
    Returns:
        A 1x7 numpy array [px py pz qx qy qz qw].
    """
    pq = np.zeros(7)
    pq[:3] = pose[:3, 3]
    rotation = Rotation.from_matrix(pose[:3, :3])
    pq[3:] = rotation.as_quat()
    return pq


def Pq_from_T_list(poses):
    pq_list = []
    for T in poses:
        pq_list.append(Pq_from_T(T))
    return pq_list


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
                C0_T_Cjpi = C0_T_Cj @ Cj_T_Cjpi
                traj.append(C0_T_Cjpi)
        else:
            # Handle the intermediate subsequences
            Cj_T_Cjp1 = poses[j, 1, :, :]  # Get the pose matrix for the first step in the subsequence
            # Append the last row [0, 0, 0, 1] to make it 4x4
            Cj_T_Cjp1 = np.vstack((Cj_T_Cjp1, np.array([0, 0, 0, 1])))
            traj.append(traj[-1] @ Cj_T_Cjp1)

    # Assert that the trajectory length matches the number of images
    assert len(traj) == imgs
    traj_pq = []
    for pose in traj:
        pq = Pq_from_T(pose)
        traj_pq.append(pq)
    return traj_pq


def save_tum_poses(times, traj_pq, outfile):
    assert len(times) == len(traj_pq), f'Inconsistent times {len(times)} and traj {len(traj_pq)}'
    with open(outfile, 'w') as s:
        for p, pq in enumerate(traj_pq):
            formatted_values = f"{times[p]:.09f} {pq[0]:.06f} {pq[1]:.06f} {pq[2]:.06f} " \
                   f"{pq[3]:.09f} {pq[4]:.09f} {pq[5]:.09f} {pq[6]:.09f}\n"
            s.write(formatted_values)
