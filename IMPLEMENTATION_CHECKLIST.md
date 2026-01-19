# Blenderbot + COMET 구현 체크리스트

## ✅ 완료된 작업

### 1. 코드 등록
- [x] `inputters/__init__.py`에 `vanilla_in_re` 추가
- [x] `CONFIG/vanilla_in_re.json` 생성
- [x] COMET 모델 설정 확인 (`inputters/vanilla_in_re.py` 라인 17-21)

### 2. 스크립트 작성 & 경로 수정
- [x] `scripts/prepare_vanilla_in_re.sh` (TRAIN_FILE: `_reformat/train.txt`)
- [x] `scripts/train_vanilla_in_re.sh` (EVAL_FILE: `_reformat/valid.txt`)
- [x] `scripts/infer_vanilla_in_re.sh` (INFER_FILE: `_reformat/test.txt`)
- [x] 스크립트 실행 권한 부여 (chmod +x)

### 3. 문서 작성
- [x] `VANILLA_IN_RE_GUIDE.md` (상세 가이드)
- [x] `QUICK_START.md` (빠른 시작 + 트러블슈팅)
- [x] `IMPLEMENTATION_CHECKLIST.md` (본 파일)

---

## 🚀 실행 단계별 체크리스트

### Step -1: 사전 훈련 모델 준비 (필수!)

#### Blenderbot 모델 다운로드
```bash
cd Blenderbot_small-90M
wget -O pytorch_model.bin https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin
cd ..
```

체크 항목:
- [ ] `Blenderbot_small-90M/pytorch_model.bin` 파일 존재 (약 370MB)
- [ ] 파일 크기 확인: `ls -lh Blenderbot_small-90M/pytorch_model.bin`

#### COMET-distill 모델 다운로드

```bash
python << 'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

os.makedirs("/data/pretrained_models/comet-distill-tokenizer", exist_ok=True)
os.makedirs("/data/pretrained_models/comet-distill-high", exist_ok=True)

print("Downloading COMET tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("allenai/comet-distill")
tokenizer.save_pretrained("/data/pretrained_models/comet-distill-tokenizer")

print("Downloading COMET model...")
model = GPT2LMHeadModel.from_pretrained("allenai/comet-distill")
model.save_pretrained("/data/pretrained_models/comet-distill-high")

print("✓ Done!")
EOF
```

체크 항목:
- [ ] `/data/pretrained_models/comet-distill-tokenizer/` 디렉토리 존재
- [ ] `/data/pretrained_models/comet-distill-high/` 디렉토리 존재
- [ ] 각 디렉토리에 `pytorch_model.bin` 또는 유사 파일 존재
- [ ] 다운로드 크기: 각 모델 ~350-400MB

예상 시간: 10-20분 (인터넷 속도에 따라)

---
```bash
cd _reformat
python process.py
```
체크 항목:
- [ ] ESConv.json 존재 (8.6MB)
- [ ] train.txt 생성됨 (70% 데이터)
- [ ] valid.txt 생성됨 (15% 데이터)
- [ ] test.txt 생성됨 (15% 데이터)
- [ ] 각 파일이 JSON Lines 형식 (각 라인이 JSON 객체)

### Step 1: COMET으로 의도/감정 추출
```bash
./scripts/prepare_vanilla_in_re.sh
```
체크 항목:
- [ ] CUDA 가용 여부 확인 (GPU 필수)
- [ ] COMET 모델 경로 존재 확인:
  - `/data/pretrained_models/comet-distill-tokenizer`
  - `/data/pretrained_models/comet-distill-high`
- [ ] 스크립트 실행 중 오류 없음
- [ ] DATA/vanilla_in_re.vanilla_in_re/data.pkl 생성됨
- [ ] meta.json 생성됨
- [ ] tokenizer.pt 생성됨

예상 시간: 수 시간 (데이터 크기와 GPU에 따라 다름)

### Step 2: 모델 훈련
```bash
./scripts/train_vanilla_in_re.sh
```
체크 항목:
- [ ] Blenderbot 모델 로드 성공
- [ ] GPU 메모리 충분 (최소 16GB)
- [ ] 훈련 시작 (epoch 1/5)
- [ ] train_log.csv 생성됨
- [ ] eval_log.csv 생성됨
- [ ] 체크포인트 저장됨 (epoch-0.bin, epoch-1.bin, ...)
- [ ] 최종 체크포인트: epoch-4.bin

예상 시간: 2-4시간 (V100, 5 epoch 기준)

### Step 3: 모델 추론
```bash
# 먼저 스크립트에서 체크포인트 경로 수정
vi scripts/infer_vanilla_in_re.sh
# CHECKPOINT_PATH="./DATA/vanilla_in_re.vanilla_in_re/{TIMESTAMP}/epoch-4.bin"

./scripts/infer_vanilla_in_re.sh
```
체크 항목:
- [ ] 체크포인트 경로 정확함
- [ ] 테스트 데이터 경로 정확함 (_reformat/test.txt)
- [ ] 추론 시작
- [ ] gen.json 생성됨 (생성 결과)
- [ ] gen.txt 생성됨
- [ ] metric.json 생성됨 (평가 메트릭)
- [ ] 결과 파일들이 res_* 폴더에 저장됨

예상 시간: ~1시간 (COMET 추론 + 생성)

---

## 📊 성능 평가

### 생성된 메트릭 확인
결과 위치: `./DATA/vanilla_in_re.vanilla_in_re/{TIMESTAMP}/epoch-4.bin/res_*/metric.json`

주요 메트릭:
```json
{
  "perplexity": 35.5,      # 낮을수록 좋음
  "bleu_1": 0.25,          # 1-gram precision
  "bleu_2": 0.18,
  "rouge_1": 0.32,         # ROUGE-1 score
  "rouge_L": 0.28,
  "meteor": 0.22
}
```

---

## 🔧 문제 해결

### 문제: COMET 모델을 찾을 수 없음
**해결**:
```python
# inputters/vanilla_in_re.py 라인 17-21 확인
tokenizer_gpt = GPT2Tokenizer.from_pretrained('/data/pretrained_models/comet-distill-tokenizer')
model_gpt = GPT2LMHeadModel.from_pretrained('/data/pretrained_models/comet-distill-high').cuda()
```
실제 경로로 수정하거나, 모델 다운로드:
```bash
# HuggingFace에서 다운로드
huggingface-cli download allenai/comet-distill
```

### 문제: CUDA Out of Memory
**해결**:
```bash
# train_vanilla_in_re.sh 수정
TRAIN_BATCH_SIZE=8          # 기본: 16
GRADIENT_ACC_STEPS=2        # 기본: 1
```

### 문제: 전처리가 너무 느림
**원인**: COMET 추론이 병목
**해결**:
- GPU 성능 확인 (nvidia-smi)
- 더 강력한 GPU로 전환
- 데이터 크기 줄여서 테스트

### 문제: 데이터 로딩 오류
**확인 사항**:
- JSON Lines 형식: 각 라인이 완전한 JSON 객체
- speaker 필드: 정확히 'usr' 또는 'sys'
- dialog 배열: 최소 2개 이상의 대화

---

## 📝 논문 대비 설정 확인

### GLHG 논문 (섹션 4.2) 기준

| 항목 | 논문 값 | 설정 | 상태 |
|------|--------|------|------|
| 모델 | Blenderbot + COMET | ✓ | ✅ |
| Optimizer | AdamW (β1=0.9, β2=0.99) | ✓ | ✅ |
| Learning Rate | 3e-5 | `scripts/train_vanilla_in_re.sh` L25 | ✅ |
| Warmup Steps | 100 | `scripts/train_vanilla_in_re.sh` L26 | ✅ |
| Batch Size | 16 | `scripts/train_vanilla_in_re.sh` L27 | ✅ |
| Max Input Length | 128 | `scripts/train_vanilla_in_re.sh` L34 | ✅ |
| Epochs | 5 | `scripts/train_vanilla_in_re.sh` L30 | ✅ |
| Infer Batch Size | 1 | `scripts/infer_vanilla_in_re.sh` L26 | ✅ |
| Max Decoding Steps | 40 | `scripts/infer_vanilla_in_re.sh` L27 | ✅ |

---

## 🎯 다음 단계

### 1. 실험 실행 및 결과 기록
- [ ] Step 0: ESConv 처리 완료
- [ ] Step 1: COMET 추출 완료 (data.pkl 크기 확인)
- [ ] Step 2: 훈련 완료 (loss 감소 추이 확인)
- [ ] Step 3: 추론 완료 (메트릭 기록)

### 2. 결과 분석
- [ ] Blenderbot + COMET 성능 평가
- [ ] GLHG 논문 원본 성과와 비교
- [ ] COMET이 성능에 미친 영향 분석

### 3. 향후 개선 (선택사항)
- [ ] Hierarchical Graph Reasoner 추가
- [ ] 다른 COMET 관계 (xWant, xNeed, xAttr 등) 실험
- [ ] Strategy modeling 통합

---

## 📚 참고 자료

- 가이드: `VANILLA_IN_RE_GUIDE.md`
- 빠른 시작: `QUICK_START.md`
- 코드: `inputters/vanilla_in_re.py` (주요 로직)
- 데이터 처리: `_reformat/process.py` (ESConv 전처리)

---

**마지막 확인**: 모든 경로가 올바른지, 스크립트가 실행 가능한지 최종 확인하세요!
