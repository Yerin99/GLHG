# Blenderbot + COMET 파이프라인 핵심 요약

## ⚠️ 중요: COMET 처리 타이밍

vanilla_in_re은 COMET을 사용하므로 **미리 처리(Pre-processing)** 해야 합니다:

| 단계 | 데이터 형식 | COMET 처리 | 설명 |
|------|-----------|----------|------|
| **Train** | `.pkl` | ✅ 사전 처리 | `prepare.py` 실행 시 완료 |
| **Valid** | `.pkl` | ✅ 사전 처리 | `prepare.py` 실행 시 완료 (새로 추가!) |
| **Test** | `.pkl` | ✅ 사전 처리 | `prepare.py` 실행 시 완료 (새로 추가!) |

**왜 미리 처리?**
- ❌ 만약 JSON Lines 그대로 사용 → 훈련 중 매 epoch마다 COMET 재실행 (수십 시간 낭비)
- ✅ 미리 pickle로 저장 → 훈련/추론 빠름 (수 시간 절약)

---

## 📊 데이터 흐름도

```
ESConv.json (원본 데이터)
    ↓ [_reformat/process.py]
train.txt, valid.txt, test.txt (JSON Lines)
    ↓ [scripts/prepare_vanilla_in_re.sh 실행]
    ├─ 1단계: train.txt 처리 (COMET 실행)
    │  └─ DATA/vanilla_in_re.vanilla_in_re/data.pkl
    ├─ 2단계: valid.txt 처리 (COMET 실행) ← NEW!
    │  └─ DATA/vanilla_in_re_valid.vanilla_in_re/data.pkl
    └─ 3단계: test.txt 처리 (COMET 실행) ← NEW!
       └─ DATA/vanilla_in_re_test.vanilla_in_re/data.pkl

   각 단계마다:
   ├─ JSON Lines 읽기
   ├─ 각 dialogue마다:
   │  ├─ 마지막 user 발화 추출
   │  ├─ COMET xIntent 생성 (사용자 의도)
   │  ├─ COMET xReact 생성 (감정 반응)
   │  └─ 컨텍스트 = [대화히스토리] + [의도] + [감정]
   └─ InputFeatures → pickle 저장
    ↓
train.py (훈련)
    ├─ Train pickle 로드 (미리 처리됨)
    ├─ Valid pickle 로드 (미리 처리됨)
    └─ 모델 훈련 및 검증
    ↓
./DATA/vanilla_in_re.vanilla_in_re/{TIMESTAMP}/epoch-4.bin
    ↓
infer.py (추론)
    ├─ 체크포인트 로드
    ├─ Test pickle 로드 (미리 처리됨)
    └─ 결과 생성 (gen.json, metric.json)
```

---

## 🚀 실행 명령어

### Step 0: 사전 훈련 모델 준비 (필수!)

**Blenderbot 다운로드:**
```bash
cd Blenderbot_small-90M
wget -O pytorch_model.bin https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin
cd ..
```

**COMET-distill 다운로드:**
```bash
python << 'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

os.makedirs("/data/pretrained_models/comet-distill-tokenizer", exist_ok=True)
os.makedirs("/data/pretrained_models/comet-distill-high", exist_ok=True)

tokenizer = GPT2Tokenizer.from_pretrained("allenai/comet-distill")
tokenizer.save_pretrained("/data/pretrained_models/comet-distill-tokenizer")

model = GPT2LMHeadModel.from_pretrained("allenai/comet-distill")
model.save_pretrained("/data/pretrained_models/comet-distill-high")

print("✓ COMET models ready!")
EOF
```

---

### Step 1: ESConv 처리 (선택사항)
```bash
cd _reformat
python process.py
```
출력: `train.txt`, `valid.txt`, `test.txt` (70/15/15 split)

### Step 2: COMET 사전 처리 (필수!) ⭐

```bash
./scripts/prepare_vanilla_in_re.sh
```

**이 스크립트가 하는 일:**
1. `train.txt` → COMET 처리 → `DATA/vanilla_in_re.vanilla_in_re/data.pkl`
2. `valid.txt` → COMET 처리 → `DATA/vanilla_in_re_valid.vanilla_in_re/data.pkl` ← NEW
3. `test.txt` → COMET 처리 → `DATA/vanilla_in_re_test.vanilla_in_re/data.pkl` ← NEW

**소요 시간**: 수 시간 (데이터 크기와 GPU에 따라 다름)

### Step 3: 모델 훈련

```bash
./scripts/train_vanilla_in_re.sh
```

- Train: 미리 처리된 pickle 로드 ✅ (빠름)
- Valid: 미리 처리된 pickle 로드 ✅ (빠름)

**소요 시간**: 2-4시간 (V100, 5 epoch)

### Step 4: 모델 추론

```bash
# 먼저 스크립트에서 체크포인트 경로 수정
vi scripts/infer_vanilla_in_re.sh

./scripts/infer_vanilla_in_re.sh
```

- Test: 미리 처리된 pickle 로드 ✅ (빠름)

**소요 시간**: ~1시간

---

## 📋 핵심 코드 분석

### Train 단계 (`train.py` L162-177)
```python
train_dataloader = inputter.train_dataloader(
    toker=toker,
    feature_dataset=inputter.train_dataset,  # ← FeatureDataset (pickle)
    batch_size=args.train_batch_size,
)
```
✅ Pickle 파일에서 미리 준비된 feature 로드

### Valid 단계 (원래 구조)
```python
eval_dataloader_loss = inputter.valid_dataloader(
    corpus_file=args.eval_input_file,  # ← JSON Lines
    batch_size=args.eval_batch_size,
)
```
❌ JSON Lines에서 런타임에 COMET 실행 (느림!)

→ **해결방법**: Valid 데이터도 미리 pickle로 처리!

### Infer 단계 (원래 구조)
```python
infer_dataloader = inputter.infer_dataloader(
    infer_input_file,  # ← JSON Lines
    toker,
)
```
❌ JSON Lines에서 런타임에 COMET 실행 (느림!)

→ **해결방법**: Test 데이터도 미리 pickle로 처리!

---

## 💡 자세한 분석

**더 자세한 내용**: `COMET_TIMING_ANALYSIS.md` 참고

핵심:
- vanilla_in_re의 `convert_data_to_inputs` 함수는 COMET을 실행
- JSON Lines 형식 데이터를 런타임에 처리하면 반복 실행됨
- 따라서 Train, Valid, Test 모두 미리 pickle로 처리 권장

---

## 🎯 처리 시간 비교

### ❌ 나쁜 방식 (현재 구조)
```
Step 1: prepare.py (train.txt만) → ~2-4시간
Step 2: train.py
  └─ 매 epoch마다 valid.txt에서 COMET 실행 → ~5시간 × 5 epoch
  └─ 추가 시간: 5-25시간 ❌
Step 3: infer.py
  └─ test.txt에서 COMET 실행 → ~1-2시간
Total: 25-30시간+
```

### ✅ 좋은 방식 (수정된 구조)
```
Step 1: prepare.py (train.txt, valid.txt, test.txt) → ~6-12시간
Step 2: train.py (pickle만 로드) → ~2-4시간
Step 3: infer.py (pickle만 로드) → ~1시간
Total: 10-17시간 ✅
```

**시간 절약: 50% 이상!**

---

## ⚠️ 주의사항

1. **GPU 필수**: COMET 처리 중 GPU 사용
2. **시간이 김**: 전체 전처리에 수 시간 소요
3. **디스크 공간**: 3개의 pickle 파일 필요
4. **경로 설정**: 스크립트의 경로 확인 필수

---

## 📚 참고 문서

- **상세 분석**: `COMET_TIMING_ANALYSIS.md`
- **전체 가이드**: `VANILLA_IN_RE_GUIDE.md`
- **체크리스트**: `IMPLEMENTATION_CHECKLIST.md`
