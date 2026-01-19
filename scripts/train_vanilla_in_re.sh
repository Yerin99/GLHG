#!/bin/bash
# 훈련 스크립트: Blenderbot + COMET (vanilla_in_re)
# GLHG 논문 4.2 Experimental Setting 기반
# - learning rate: 3e-5
# - warmup steps: 100
# - batch size: 16
# - epochs: 5
# - max input length: 128

# ============================================================================
# 설정 변수 (사용자 환경에 맞게 수정 필요)
# ============================================================================

# GPU 설정
export CUDA_VISIBLE_DEVICES=0

# Inputter 및 Config 설정
INPUTTER_NAME="vanilla_in_re"
CONFIG_NAME="vanilla_in_re"

# 데이터 파일 경로
# ESConv 데이터셋을 process.py로 처리한 결과 파일 경로로 수정하세요
# 예: ./_reformat/valid.txt
EVAL_FILE="./_reformat/valid.txt"

# 학습 하이퍼파라미터 (GLHG 논문 기준)
LEARNING_RATE=3e-5          # 논문: 3e-5
WARMUP_STEPS=100            # 논문: 100 steps
TRAIN_BATCH_SIZE=16         # 논문: 16
EVAL_BATCH_SIZE=8
GRADIENT_ACC_STEPS=1        # 필요시 조정
NUM_EPOCHS=5                # 논문: 5 epochs
SEED=42

# 최대 길이 설정
MAX_INPUT_LENGTH=128        # 논문: 128
MAX_DECODER_INPUT_LENGTH=50

# 검증 주기 (옵션)
# NUM_EPOCHS를 사용할 경우 주석 처리
# VALID_STEP=2000
# NUM_OPTIM_STEPS=20000

# ============================================================================
# 훈련 실행
# ============================================================================

echo "=================================================="
echo "훈련 시작: Blenderbot + COMET (${INPUTTER_NAME})"
echo "=================================================="
echo "Inputter: ${INPUTTER_NAME}"
echo "Config: ${CONFIG_NAME}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${TRAIN_BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "Seed: ${SEED}"
echo "=================================================="

python train.py \
    --config_name ${CONFIG_NAME} \
    --inputter_name ${INPUTTER_NAME} \
    --seed ${SEED} \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_decoder_input_length ${MAX_DECODER_INPUT_LENGTH} \
    --eval_input_file ${EVAL_FILE} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC_STEPS} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --warmup_steps ${WARMUP_STEPS} \
    --num_epochs ${NUM_EPOCHS} \
    --max_grad_norm 1.0 \
    --fp16 False \
    --pbar True

# 에포크 기반이 아닌 step 기반 훈련을 원할 경우:
# --num_optim_steps 대신 사용하고 --num_epochs 제거
# --num_optim_steps ${NUM_OPTIM_STEPS} \
# --valid_step ${VALID_STEP} \

echo "=================================================="
echo "훈련 완료!"
echo "모델 저장 위치: ./DATA/${INPUTTER_NAME}.${CONFIG_NAME}/"
echo "=================================================="
