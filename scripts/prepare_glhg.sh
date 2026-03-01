#!/bin/bash
# GLHG 데이터 전처리 스크립트
#
# 이 스크립트는 COMET을 사용하여 xIntent를 생성하므로
# GPU가 필요하며 시간이 오래 걸릴 수 있습니다.

set -e

# Conda 환경 활성화
source /home/yerin/miniconda3/etc/profile.d/conda.sh
conda activate glhg

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 설정
CONFIG_NAME="glhg"
INPUTTER_NAME="glhg"
TRAIN_INPUT_FILE="./_reformat/train.txt"
MAX_INPUT_LENGTH=128
MAX_DECODER_INPUT_LENGTH=50

echo "========================================"
echo "GLHG Data Preparation"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Inputter: ${INPUTTER_NAME}"
echo "Train File: ${TRAIN_INPUT_FILE}"
echo "Max Input Length: ${MAX_INPUT_LENGTH}"
echo "Max Decoder Input Length: ${MAX_DECODER_INPUT_LENGTH}"
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

cd /home/yerin/baseline/GLHG

# 데이터 전처리 실행
# single_processing 옵션을 사용하여 CUDA 충돌 방지
python prepare.py \
    --config_name ${CONFIG_NAME} \
    --inputter_name ${INPUTTER_NAME} \
    --train_input_file ${TRAIN_INPUT_FILE} \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_decoder_input_length ${MAX_DECODER_INPUT_LENGTH} \
    --single_processing \
    2>&1 | tee prepare_glhg.log

echo "========================================"
echo "Data preparation completed!"
echo "Data saved to: ./DATA/${INPUTTER_NAME}.${CONFIG_NAME}/"
echo "========================================"
