#!/bin/bash
# GLHG (Global-to-Local Hierarchical Graph Network) 추론 스크립트
#
# 논문 추론 설정:
# - Batch size: 1
# - Max decoding steps: 40
# - Top-p: 0.9
# - Temperature: 0.7

set -e

# Conda 환경 활성화
source /home/yerin/miniconda3/etc/profile.d/conda.sh
conda activate glhg

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 인자 파싱
CHECKPOINT_PATH=${1:-""}
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: $0 <checkpoint_path> [output_dir]"
    echo "Example: $0 ./DATA/glhg.glhg/2026-01-20.../epoch-4.bin"
    exit 1
fi

OUTPUT_DIR=${2:-$(dirname "$CHECKPOINT_PATH")}

# 추론 설정 (논문 기준)
CONFIG_NAME="glhg"
INPUTTER_NAME="glhg"
INFER_INPUT_FILE="./_reformat/test.txt"
MAX_INPUT_LENGTH=128
MAX_DECODER_INPUT_LENGTH=50
INFER_BATCH_SIZE=1
MAX_LENGTH=40
TOP_P=0.9
TOP_K=0
TEMPERATURE=0.7
NUM_BEAMS=1
LENGTH_PENALTY=1.0
REPETITION_PENALTY=1.0
NO_REPEAT_NGRAM_SIZE=3

echo "========================================"
echo "GLHG Inference Configuration"
echo "========================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_NAME}"
echo "Inputter: ${INPUTTER_NAME}"
echo "Test File: ${INFER_INPUT_FILE}"
echo "Max Length: ${MAX_LENGTH}"
echo "Top-p: ${TOP_P}"
echo "Temperature: ${TEMPERATURE}"
echo "CUDA Devices: ${CUDA_VISIBLE_DEVICES}"
echo "========================================"

# 추론 실행
python infer.py \
    --config_name ${CONFIG_NAME} \
    --inputter_name ${INPUTTER_NAME} \
    --load_checkpoint ${CHECKPOINT_PATH} \
    --infer_input_file ${INFER_INPUT_FILE} \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_decoder_input_length ${MAX_DECODER_INPUT_LENGTH} \
    --infer_batch_size ${INFER_BATCH_SIZE} \
    --max_length ${MAX_LENGTH} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --temperature ${TEMPERATURE} \
    --num_beams ${NUM_BEAMS} \
    --length_penalty ${LENGTH_PENALTY} \
    --repetition_penalty ${REPETITION_PENALTY} \
    --no_repeat_ngram_size ${NO_REPEAT_NGRAM_SIZE} \
    2>&1 | tee ${OUTPUT_DIR}/infer_glhg.log

echo "========================================"
echo "Inference completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"
