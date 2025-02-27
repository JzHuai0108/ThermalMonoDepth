import torch
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
# from imageio import imread

import sys
sys.path.append('./common/')
from utils.convert_poses import cat_poses, save_tum_poses, Pq_from_T_list
import models
from loss.inverse_warp import pose_vec2mat

parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height") 
parser.add_argument("--img-width", default=320, type=int, help="Image width") 
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--max-value", default=2**14, type=float, help="max pixel value in the thermal images")
parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--modality", default='Thermal', type=str, help="sensor modality, thermal or RGB")
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for testing', default=5)
parser.add_argument("--sequences", default=['indoor_aggresive_dark'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50], help='depth network architecture.')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_tensor_image(img, args):
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    img = (torch.from_numpy(img).float() / args.max_value)
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img

@torch.no_grad()
def main():
    args = parser.parse_args()

    # load models
    disp_pose_net = models.DispPoseResNet(args.resnet_layers, False, num_channel=1).to(device)
    pose_net = disp_pose_net.PoseResNet

    weights = torch.load(args.pretrained_posenet)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()

    seq_length = 5

    # load data loader
    from eval_vivid.pose_evaluation_utils import test_framework_VIVID as test_framework
    dataset_dir = Path(args.dataset_dir)
    print(f'Sensor modality {args.modality}')
    framework = test_framework(dataset_dir, args.sequences, seq_length=seq_length, step=1, modality=args.modality)

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.makedirs_p()
        predictions_array = np.zeros((len(framework), seq_length, 3, 4))

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']
        squence_imgs = []
        for i, img in enumerate(imgs):
            img = load_tensor_image(img, args)
            squence_imgs.append(img)

        global_pose = np.eye(4)
        poses = []
        poses.append(global_pose[0:3, :])

        for iter in range(seq_length - 1):
            pose = pose_net(squence_imgs[iter], squence_imgs[iter + 1])
            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()

            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])]) # C(j+i)_T_C(j+i-1)
            global_pose = global_pose @  np.linalg.inv(pose_mat) # C(j)_T_C(j+i)
            poses.append(global_pose[0:3, :])

        final_poses = np.stack(poses, axis=0)

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE', 'RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))

    if args.output_dir is not None:
        np.save(output_dir/'predictions.npy', predictions_array)
        if len(args.sequences) < 2:
            print(f'Saving stamped_traj_estimate0.txt under {output_dir}')
            traj = cat_poses(predictions_array)
            img_times = framework.times[0]
            save_tum_poses(img_times, traj, output_dir/'stamped_traj_estimate0.txt')
            poses = framework.poses[0]
            gt_traj = Pq_from_T_list(poses)
            assert len(gt_traj) == len(traj), f"Inconsistent traj poses {len(traj)} and gt poses {len(gt_traj)}"
            print(f'Saving stamped_groundtruth.txt under {output_dir}')
            save_tum_poses(img_times, gt_traj, output_dir/'stamped_groundtruth.txt')


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:, :, -1] * pred[:, :, -1])/np.sum(pred[:, :, -1] ** 2)
    ATE = np.linalg.norm((gt[:, :, -1] - scale_factor * pred[:, :, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:, :3] @ np.linalg.inv(pred_pose[:, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length


def compute_pose(pose_net, tgt_img, ref_imgs):
    poses = []
    for ref_img in ref_imgs:
        pose = pose_net(tgt_img, ref_img).unsqueeze(1)
        poses.append(pose)
    poses = torch.cat(poses, 1)
    return poses


if __name__ == '__main__':
    main()
