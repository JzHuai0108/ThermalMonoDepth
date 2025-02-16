#!/bin/bash
# run script : bash test_vivid_outdoor.sh

GPU_ID=1
DATA_ROOT=datasets/VIVID_256/
RESULTS_DIR=results

RESNET=18
IMG_H=256
IMG_W=320
DATASET=VIVID
INPUT_TYPE=T 
DEPTH_GT_DIR=Depth_T 
POSE_GT=poses_T.txt 

# outdoor testset
process_modality() {
modality=$1
NAMES=("T_vivid_resnet18_outdoor")
for NAME in ${NAMES[@]}; do
	SEQS=("outdoor_robust_night1" "outdoor_robust_night2"
	      "outdoor_robust_day1" "outdoor_robust_day2")
	POSE_NET=checkpoints/${NAME}/exp_pose_pose_model_best.pth.tar
	DISP_NET=checkpoints/${NAME}/dispnet_disp_model_best.pth.tar
	echo "${NAME}"

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
		--img-height $IMG_H --img-width $IMG_W --max-value 16384 \
		--dataset-dir ${DATA_DIR} --output-dir $OUTPUT_DEPTH_DIR 2>&1 | tee ${OUTPUT_DEPTH_DIR}/disp.txt

		CUDA_VISIBLE_DEVICES=${GPU_ID} python -u eval_vivid/eval_depth.py \
		--dataset $DATASET --pred_depth ${OUTPUT_DEPTH_DIR}/predictions.npy \
		--gt_depth ${DATA_DIR}/${DEPTH_GT_DIR}  --scene outdoor 2>&1 | tee ${OUTPUT_DEPTH_DIR}/eval_depth.txt 

#		rm ${OUTPUT_DEPTH_DIR}/predictions.npy


		# Pose Evaulation 
		CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test_pose.py \
		--resnet-layers $RESNET --pretrained-posenet $POSE_NET \
		--img-height $IMG_H --img-width $IMG_W --max-value 16384 \
		--dataset-dir ${DATA_ROOT} --output-dir ${OUTPUT_POSE_DIR} \
		--sequences ${SEQ} 2>&1 | tee ${OUTPUT_POSE_DIR}/eval_pose.txt
	done
done
}

process_modality RGB
process_modality Thermal
