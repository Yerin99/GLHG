#!/bin/bash
# 데이터 전처리 스크립트: Blenderbot + COMET (vanilla_in_re)
# 
# 현재 구조: Train만 사전 처리, Valid/Test는 런타임 처리
# (COMET_TIMING_ANALYSIS.md의 방안 B)

# ============================================================================
# 설정 변수
# ============================================================================

INPUTTER_NAME="vanilla_in_re"
CONFIG_NAME="vanilla_in_re"

# 최대 길이 설정 (논문: max_input_length=128)
MAX_INPUT_LENGTH=128
MAX_DECODER_INPUT_LENGTH=50

# ============================================================================
# 전처리 실행 (Train 데이터만)
# ============================================================================

echo "=================================================="
echo "데이터 전처리 시작: ${INPUTTER_NAME}"
echo "Train 데이터만 처리합니다 (약 2-4시간 소요)"
echo "=================================================="
echo ""

python prepare.py \
    --config_name ${CONFIG_NAME} \
    --inputter_name ${INPUTTER_NAME} \
    --train_input_file _reformat/train.txt \
    --max_input_length ${MAX_INPUT_LENGTH} \
    --max_decoder_input_length ${MAX_DECODER_INPUT_LENGTH} \
    --single_processing

if [ $? -ne 0 ]; then
    echo "Train 데이터 전처리 실패!"
    exit 1
fi

echo ""
echo "=================================================="
echo "✅ Train 데이터 전처리 완료!"
echo ""
echo "생성된 파일:"
echo "  ./DATA/${CONFIG_NAME}.${CONFIG_NAME}/data.pkl"
echo ""
echo "다음 단계: train_vanilla_in_re.sh 실행"
echo "=================================================="

