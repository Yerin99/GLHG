#!/bin/bash
# GLHG (Global-to-Local Hierarchical Graph Network) 훈련 스크립트
#
# 논문 설정:
# - Learning rate: 3e-5
# - Batch size: 16
# - Epochs: 5
# - Warmup steps: 100
# - λ1 = λ2 = 0.5

set -e

# Conda 환경 활성화
source /home/yerin/miniconda3/etc/profile.d/conda.sh
conda activate glhg

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 하이퍼파라미터 (논문 기준)
LEARNING_RATE=${LEARNING_RATE:-3e-5}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-5}
WARMUP_STEPS=${WARMUP_STEPS:-100}
MAX_INPUT_LENGTH=${MAX_INPUT_LENGTH:-128}
MAX_DECODER_INPUT_LENGTH=${MAX_DECODER_INPUT_LENGTH:-50}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
SEED=${SEED:-42}

# 경로 설정
CONFIG_NAME="glhg"
INPUTTER_NAME="glhg"
EVAL_INPUT_FILE="./_reformat/valid.txt"

echo "========================================"
echo "GLHG Training Configuration"
echo "========================================"
echo "Config: ${CONFIG_NAME}"
echo "Inputter: ${INPUTTER_NAME}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "Max Input Length: ${MAX_INPUT_LENGTH}"
echo "Max Decoder Input Length: ${MAX_DECODER_INPUT_LENGTH}"
echo "Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Seed: ${SEED}"
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# 훈련 실행
python train.py \
    --config_name ${CONFIG_NAME} \
    --inputter_name ${INPUTTER_NAME} \
    --seed ${SEED} \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_decoder_input_length ${MAX_DECODER_INPUT_LENGTH} \
    --train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --eval_batch_size 8 \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --num_epochs ${NUM_EPOCHS} \
    --max_grad_norm 1.0 \
    --eval_input_file ${EVAL_INPUT_FILE} \
    --fp16 false \
    --pbar true \
    2>&1 | tee train_glhg.log

echo "========================================"
echo "Training completed!"
echo "========================================"
