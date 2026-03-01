#!/bin/bash
# 추론 스크립트: Blenderbot + COMET (vanilla_in_re)
# GLHG 논문 4.2 Experimental Setting 기반
# - batch size: 1 (논문)
# - max decoding steps: 40 (논문)

# ============================================================================
# 설정 변수 (사용자 환경에 맞게 수정 필요)
# ============================================================================

# GPU 설정
export CUDA_VISIBLE_DEVICES=0

# Inputter 및 Config 설정
INPUTTER_NAME="vanilla_in_re"
CONFIG_NAME="vanilla_in_re"

# 체크포인트 경로 (epoch-1: 가장 낮은 PPL 16.73)
CHECKPOINT_PATH="./DATA/vanilla_in_re.vanilla_in_re/2026-01-19180015.3e-05.16.1gpu/epoch-1.bin"

# 추론 데이터 파일 경로
# ESConv 데이터셋을 process.py로 처리한 결과 파일 경로로 수정하세요
# 예: ./_reformat/test.txt
INFER_FILE="./_reformat/test.txt"

# 추론 하이퍼파라미터 (infer_bos.out 기준)
INFER_BATCH_SIZE=1          # 논문: batch size = 1 (infer_bos.out은 16이지만 논문 따름)
MAX_LENGTH=40               # infer_bos.out 기준
MIN_LENGTH=10               # infer_bos.out 기준
MAX_INPUT_LENGTH=128        # 논문: max length of input sequence = 128
MAX_DECODER_INPUT_LENGTH=50
SEED=42

# 생성 전략 설정 (infer_bos.out 기준)
NUM_BEAMS=1                 # infer_bos.out
TOP_K=0                     # infer_bos.out
TOP_P=0.9                   # infer_bos.out (nucleus sampling)
TEMPERATURE=0.7             # infer_bos.out
LENGTH_PENALTY=1.0          # infer_bos.out
REPETITION_PENALTY=1.0      # infer_bos.out
NO_REPEAT_NGRAM=3           # infer_bos.out

# ============================================================================
# 추론 실행
# ============================================================================

echo "=================================================="
echo "추론 시작: Blenderbot + COMET (${INPUTTER_NAME})"
echo "=================================================="
echo "Inputter: ${INPUTTER_NAME}"
echo "Config: ${CONFIG_NAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Infer file: ${INFER_FILE}"
echo "Batch size: ${INFER_BATCH_SIZE}"
echo "Max length: ${MAX_LENGTH}"
echo "Num beams: ${NUM_BEAMS}"
echo "=================================================="

python infer.py \
    --config_name ${CONFIG_NAME} \
    --inputter_name ${INPUTTER_NAME} \
    --seed ${SEED} \
    --load_checkpoint ${CHECKPOINT_PATH} \
    --fp16 False \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_decoder_input_length ${MAX_DECODER_INPUT_LENGTH} \
    --min_length ${MIN_LENGTH} \
    --max_length ${MAX_LENGTH} \
    --infer_batch_size ${INFER_BATCH_SIZE} \
    --infer_input_file ${INFER_FILE} \
    --temperature ${TEMPERATURE} \
    --top_k ${TOP_K} \
    --top_p ${TOP_P} \
    --num_beams ${NUM_BEAMS} \
    --length_penalty ${LENGTH_PENALTY} \
    --repetition_penalty ${REPETITION_PENALTY} \
    --no_repeat_ngram_size ${NO_REPEAT_NGRAM} \
    --num_return_sequences 1

echo "=================================================="
echo "추론 완료!"
echo "결과 저장 위치: 체크포인트 디렉토리 내 res_* 폴더"
echo "=================================================="

# ============================================================================
# 다양한 디코딩 전략 예시
# ============================================================================

# 1. Greedy Decoding (가장 빠름)
# --num_beams 1 --top_k 0 --top_p 1.0

# 2. Beam Search (더 나은 품질)
# --num_beams 5 --top_k 0 --top_p 1.0

# 3. Top-k Sampling (다양성)
# --num_beams 1 --top_k 50 --top_p 1.0 --temperature 0.7

# 4. Nucleus Sampling (균형)
# --num_beams 1 --top_k 0 --top_p 0.9 --temperature 0.8
