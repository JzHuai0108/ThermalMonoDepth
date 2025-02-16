#!/bin/bash
# run script : bash test_vivid_indoor.sh
GPU_ID=0
DATA_ROOT=datasets/irs_rtvi_datasets_2021
RESULTS_DIR=results

RESNET=18
IMG_H=256
IMG_W=320
# IMG_H=512
# IMG_W=640
DATASET=RRXIO
DEPTH_GT_DIR=Depth_T 
POSE_GT=poses_T.txt 

process_modality() {
modality=$1
NAMES=("T_vivid_resnet18_indoor")
for NAME in ${NAMES[@]}; do
	# indoor testset
	SEQS=('mocap_easy' 'gym' 'mocap_dark' 'indoor_floor' 'mocap_medium'
          'mocap_difficult' 'mocap_dark_fast')
	POSE_NET=checkpoints/${NAME}/exp_pose_pose_model_best.pth.tar
	DISP_NET=checkpoints/${NAME}/dispnet_disp_model_best.pth.tar 
	echo "${NAME}"

	# depth
	for SEQ in ${SEQS[@]}; do
		echo "Seq_name : ${SEQ}"

		#mkdir -p ${RESULTS_DIR}
		DATA_DIR=${DATA_ROOT}/${SEQ}/
		OUTPUT_DEPTH_DIR=${RESULTS_DIR}/${DATASET}/$modality/Depth/${SEQ}/
		OUTPUT_POSE_DIR=${RESULTS_DIR}/${DATASET}/$modality/${SEQ}/
		mkdir -p ${OUTPUT_DEPTH_DIR}
		mkdir -p ${OUTPUT_POSE_DIR}

		# Detph Evaulation 
		CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test_disp.py \
		--resnet-layers $RESNET --pretrained-dispnet $DISP_NET \
		--img-height $IMG_H --img-width $IMG_W --max-value 256 --modality $modality \
		--dataset-dir ${DATA_DIR} --output-dir $OUTPUT_DEPTH_DIR 2>&1 | tee ${OUTPUT_DEPTH_DIR}/disp.txt

#		rm ${OUTPUT_DEPTH_DIR}/predictions.npy

		# Pose Evaulation 
		cmd="python -u test_pose.py \
		--resnet-layers $RESNET --pretrained-posenet $POSE_NET \
		--img-height $IMG_H --img-width $IMG_W --max-value 256 --modality $modality \
		--dataset-dir ${DATA_ROOT} --output-dir ${OUTPUT_POSE_DIR} \
		--sequences ${SEQ}"
		echo $cmd
		CUDA_VISIBLE_DEVICES=${GPU_ID} $cmd 2>&1 | tee ${OUTPUT_POSE_DIR}/eval_pose.txt
	done
done
}

process_modality visual_undistort
process_modality thermal_undistort
