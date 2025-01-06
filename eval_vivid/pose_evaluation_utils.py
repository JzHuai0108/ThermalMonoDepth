import numpy as np
from path import Path
from imageio import imread
import cv2
from tqdm import tqdm


class test_framework_VIVID(object):
    def __init__(self, root, sequence_set, seq_length=3, step=1, modality='Thermal'):
        self.root = root
        self.img_files, self.poses, self.sample_indices, self.times = read_scene_data(self.root, sequence_set, seq_length, step, modality)

    def generator(self):
        for img_list, pose_list, sample_list in zip(self.img_files, self.poses, self.sample_indices):
            for snippet_indices in sample_list:
                imgs = []
                for i in snippet_indices:
                    orig_img = imread(img_list[i]).astype(np.float32)
                    if len(orig_img.shape) < 3:
                        img = np.expand_dims(orig_img, axis=2)
                    else:
                        img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
                        img = np.expand_dims(img, axis=2)
                    imgs.append(img)

                poses = np.stack([pose_list[i] for i in snippet_indices])
                first_pose = poses[0]
                poses[:,:,-1] -= first_pose[:,-1]
                compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

                yield {'imgs': imgs,
                       'path': img_list[snippet_indices[0]],
                       'poses': compensated_poses
                       }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return sum(len(samples) for samples in self.sample_indices)


def load_times(timefile):
    """
    timefile: each line, time rel_path
    """
    times = []
    with open(timefile, 'r') as s:
        for l in s:
            times.append(float(l.split()[0]))
    return times


def interpolate_poses(poses, img_times):
    """
    Interpolates poses for given image timestamps and outputs 3x4 transformation matrices.
    
    Args:
        poses (list of list): List of poses, each formatted as 
                              [time, px, py, pz, qx, qy, qz, qw].
        img_times (list of float): List of image timestamps.
    
    Returns:
        list of ndarray: List of 3x4 transformation matrices, matching the img_times.
    """
    pose_times = [pose[0] for pose in poses]
    positions = np.array([pose[1:4] for pose in poses])
    quaternions = np.array([pose[4:] for pose in poses])  # [qx, qy, qz, qw]

    from scipy.spatial.transform import Rotation as R, Slerp
    rotations = R.from_quat(quaternions)
    slerp = Slerp(pose_times, rotations)

    transformation_matrices = []
    lower_idx = 0  # Start pointer for lower bound

    for t in img_times:
        if t <= pose_times[0]:
            translation = positions[0]
            rotation_matrix = R.from_quat(quaternions[0]).as_matrix()
            transformation_matrices.append(create_transformation_matrix(rotation_matrix, translation))
            continue

        if t >= pose_times[-1]:
            translation = positions[-1]
            rotation_matrix = R.from_quat(quaternions[-1]).as_matrix()
            transformation_matrices.append(create_transformation_matrix(rotation_matrix, translation))
            continue

        # Increment lower_idx until it finds the correct interval
        while lower_idx < len(pose_times) - 1 and pose_times[lower_idx + 1] <= t:
            lower_idx += 1

        t1, t2 = pose_times[lower_idx], pose_times[lower_idx + 1]
        alpha = (t - t1) / (t2 - t1)

        p1, p2 = positions[lower_idx], positions[lower_idx + 1]
        interp_position = (1 - alpha) * p1 + alpha * p2

        interp_rotation = slerp([t]).as_matrix()[0]
        transformation_matrices.append(create_transformation_matrix(interp_rotation, interp_position))

    return transformation_matrices


def create_transformation_matrix(rotation_matrix, translation_vector):
    """
    Creates a 3x4 transformation matrix from a rotation matrix and translation vector.
    
    Args:
        rotation_matrix (ndarray): A 3x3 rotation matrix.
        translation_vector (ndarray): A 1x3 translation vector.
    
    Returns:
        ndarray: A 3x4 transformation matrix.
    """
    transformation_matrix = np.zeros((3, 4))
    transformation_matrix[:, :3] = rotation_matrix
    transformation_matrix[:, 3] = translation_vector
    return transformation_matrix


def read_scene_data(data_root, sequence_set, seq_length=3, step=1, modality='Thermal'):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    times_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

    sequences = []
    for seq in sequence_set:
        sequences.append((data_root/seq))

    print('getting test metadata for theses sequences : {}'.format(sequences))
    for sequence in tqdm(sequences):
        if modality == 'Thermal': # vivid 256 thermal
            imgs = sorted((sequence/'Thermal').files('*.png'))
            poses = np.genfromtxt(sequence/'poses_T.txt').astype(np.float64).reshape(-1, 3, 4)
            img_times = load_times(sequence/f'{modality}.txt')
        elif modality == 'RGB': # vivid 256 RGB
            imgs = sorted((sequence/'RGB').files('*.png'))
            poses = np.genfromtxt(sequence/'poses_RGB.txt').astype(np.float64).reshape(-1, 3, 4)
            img_times = load_times(sequence/f'{modality}.txt')
        elif 'thermal' in modality.lower(): # rrxio thermal
            imgs = sorted((sequence/modality).files('*.png'))
            poses = np.genfromtxt(sequence/'gt_thermal.txt').astype(np.float64)
            img_times = load_times(sequence/f'{modality}.txt')
            assert len(img_times) == len(imgs), f'Inconsistent image times {len(img_times)} and images {len(imgs)}' 
            interp_poses = interpolate_poses(poses, img_times)
            poses = interp_poses
        elif 'visual' in modality.lower(): # rrxio visual
            imgs = sorted((sequence/modality).files('*.png'))
            poses = np.genfromtxt(sequence/'gt_visual.txt').astype(np.float64)
            img_times = load_times(sequence/f'{modality}.txt')
            assert len(img_times) == len(imgs), f'Inconsistent image times {len(img_times)} and images {len(imgs)}' 
            interp_poses = interpolate_poses(poses, img_times)
            poses = interp_poses
   
        # construct 5-snippet sequences
        tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        im_sequences.append(imgs)
        poses_sequences.append(poses)
        indices_sequences.append(snippet_indices)
        times_sequences.append(img_times)
    return im_sequences, poses_sequences, indices_sequences, times_sequences
