# COMET 처리 타이밍 분석: 사전 처리 vs 런타임

## 🔍 핵심 답변

**후자가 맞습니다!** COMET은 **사전에 처리(미리 pkl로 만들어야)** 합니다.

그리고 **train, valid, test 모두 동일하게 처리**해야 합니다!

---

## 📥 COMET 모델 준비

### 다운로드 방법

COMET-distill 모델은 [GitHub - allenai/comet-atomic-2020](https://github.com/allenai/comet-atomic-2020)에서 제공하며, HuggingFace를 통해 다운로드할 수 있습니다.

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

print("✓ COMET models ready!")
EOF
```

### 검증
```bash
ls -lh /data/pretrained_models/comet-distill-tokenizer/
ls -lh /data/pretrained_models/comet-distill-high/
```

---

## 📊 코드 흐름 분석

### 1️⃣ Train 단계 (`train.py` L162-177)

```python
# train.py L162-177
train_dataloader = inputter.train_dataloader(
    toker=toker,
    feature_dataset=inputter.train_dataset,  # ← FeatureDataset 클래스 사용
    batch_size=args.train_batch_size,
    **names
)
```

**inputter.train_dataloader는 뭘까요?**

```python
# vanilla_in_re.py L34
self.train_dataloader = BucketingDataLoader
```

BucketingDataLoader를 보면:
- **pickle 파일에서 미리 저장된 feature를 로드합니다**
- **런타임에 COMET을 실행하지 않습니다** ❌

---

### 2️⃣ Valid 단계 (`train.py` L190-195)

```python
# train.py L190-195
eval_dataloader_loss = inputter.valid_dataloader(
    toker=toker,
    corpus_file=args.eval_input_file,  # ← JSON Lines 파일!
    batch_size=args.eval_batch_size,
    **dataloader_kwargs
)
```

여기서 `valid_dataloader`는:

```python
# vanilla_in_re.py L38
self.valid_dataloader = DynamicBatchingLoader
```

DynamicBatchingLoader를 보면 (`vanilla_in_re.py` L242-292):

```python
def _iter_epoch(self):
    # L265-271
    with open(self.corpus, 'r', encoding="utf-8") as f:
        reader = f.readlines()  # JSON Lines 읽기
    
    for line in tqdm.tqdm(reader, total=len(reader)):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, self.toker, **self.kwargs)  # ← COMET 실행!
        features.extend(convert_inputs_to_features(inputs, self.toker, **self.kwargs))
        # ...
```

**검증 시마다 COMET이 런타임으로 실행됩니다!** ⚠️

---

### 3️⃣ Infer 단계 (`infer.py` L157-161)

```python
# infer.py L157-161
infer_dataloader = inputter.infer_dataloader(
    infer_input_file,
    toker,
    **dataloader_kwargs
)
```

`infer_dataloader`는:

```python
# vanilla_in_re.py L42
self.infer_dataloader = get_infer_batch
```

get_infer_batch 함수 (`vanilla_in_re.py` L303-333):

```python
def get_infer_batch(infer_input_file, toker, **kwargs):
    # L307-308
    with open(infer_input_file, 'r', encoding="utf-8") as f:
        reader = f.readlines()  # JSON Lines 읽기
    
    for sample_id, line in tqdm.tqdm(enumerate(reader)):
        data = json.loads(line)
        inputs = convert_data_to_inputs(data, toker, **kwargs)  # ← COMET 실행!
        tmp_features = convert_inputs_to_features(inputs, toker, **kwargs)
        # ...
```

**추론 시마다 COMET이 런타임으로 실행됩니다!** ⚠️

---

## 🚀 결론

| 단계 | 데이터 형식 | COMET 처리 | 타이밍 |
|------|-----------|----------|-------|
| **Train** | pickle 파일 | ✅ 미리 처리됨 | `prepare.py` 실행 시 |
| **Valid** | JSON Lines | ❌ 런타임 처리 | 훈련 중 매번 실행 |
| **Infer** | JSON Lines | ❌ 런타임 처리 | 추론 중 매번 실행 |

---

## ⚠️ 문제점 분석

### 현재 설정의 문제

1. **Train만 미리 처리** ❌
   - `scripts/prepare_vanilla_in_re.sh`는 train.txt만 처리
   - valid.txt와 test.txt는 처리하지 않음

2. **Valid와 Infer 중에 COMET 반복 실행** ❌
   - 훈련 중 매 epoch마다 valid 단계에서 COMET 재실행
   - 추론 중에도 COMET 재실행
   - **매우 느린 속도, 중복 계산**

3. **메모리 낭비** ❌
   - COMET을 여러 번 로드/언로드
   - GPU 메모리 비효율적

---

## ✅ 올바른 방식

### 방안 A: 모두 미리 처리 (권장) 🌟

```bash
# 1. Train 데이터 사전 처리
python prepare.py \
    --config_name vanilla_in_re \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/train.txt \
    --single_processing

# 2. Valid 데이터 사전 처리 (추가 필요!)
python prepare.py \
    --config_name vanilla_in_re_valid \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/valid.txt \
    --single_processing

# 3. Test 데이터 사전 처리 (추가 필요!)
python prepare.py \
    --config_name vanilla_in_re_test \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/test.txt \
    --single_processing

# 4. 훈련 실행
python train.py \
    --config_name vanilla_in_re \
    --inputter_name vanilla_in_re \
    --eval_input_file _reformat/valid.txt  # ← 여전히 JSON Lines
    # ...

# 5. 추론 실행
python infer.py \
    --config_name vanilla_in_re_test \
    --inputter_name vanilla_in_re \
    --infer_input_file _reformat/test.txt  # ← 여전히 JSON Lines
    # ...
```

**장점:**
- ✅ 미리 계산되어 있으므로 훈련/추론 빠름
- ✅ COMET 반복 실행 없음
- ✅ 재현성 보장 (동일한 COMET 결과)

**단점:**
- ❌ 디스크 공간 사용 (3개 pickle 파일)

---

### 방안 B: 런타임 처리 (현재 구조)

```bash
# 1. Train 데이터만 사전 처리
python prepare.py \
    --config_name vanilla_in_re \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/train.txt \
    --single_processing

# 2. 훈련 실행 (valid는 런타임 처리)
python train.py \
    --config_name vanilla_in_re \
    --inputter_name vanilla_in_re \
    --eval_input_file _reformat/valid.txt  # ← 런타임에 COMET 실행
    # ...

# 3. 추론 실행 (test도 런타임 처리)
python infer.py \
    --config_name vanilla_in_re \
    --inputter_name vanilla_in_re \
    --infer_input_file _reformat/test.txt  # ← 런타임에 COMET 실행
    # ...
```

**장점:**
- ✅ 디스크 공간 절약
- ✅ 한 번만 prepare.py 실행

**단점:**
- ❌ 훈련 중 매 epoch마다 COMET 재실행 (매우 느림)
- ❌ 추론 중에도 COMET 재실행 (느림)
- ❌ 재현성 문제 (매번 다른 COMET 결과 가능)

---

## 🎯 코드 수정안

### 1. `scripts/prepare_vanilla_in_re.sh` 수정

**현재**: Train만 처리
```bash
TRAIN_FILE="./_reformat/train.txt"
python prepare.py --train_input_file ${TRAIN_FILE} ...
```

**수정**: Train, Valid, Test 모두 처리
```bash
# Train
python prepare.py \
    --config_name vanilla_in_re \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/train.txt \
    --single_processing

# Valid (새로 추가)
python prepare.py \
    --config_name vanilla_in_re_valid \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/valid.txt \
    --single_processing

# Test (새로 추가)
python prepare.py \
    --config_name vanilla_in_re_test \
    --inputter_name vanilla_in_re \
    --train_input_file _reformat/test.txt \
    --single_processing
```

### 2. `QUICK_START.md` 수정

현재 부정확한 설명:
```
prepare.py (--inputter_name vanilla_in_re)
    ├─ 각 dialogue마다:
    │  ├─ 마지막 user 발화 추출
    │  ├─ COMET-distill로 xIntent 생성
    │  ├─ COMET-distill로 xReact 생성
    │  └─ 컨텍스트 = [대화히스토리] + [의도] + [감정반응]
    └─ data.pkl 저장
```

**수정 후:**
```
Step 1: prepare.py (Train, Valid, Test 각각 실행)
    ├─ train.txt → DATA/vanilla_in_re.vanilla_in_re/data.pkl
    ├─ valid.txt → DATA/vanilla_in_re_valid.vanilla_in_re/data.pkl
    └─ test.txt → DATA/vanilla_in_re_test.vanilla_in_re/data.pkl
    
    각 파일마다 COMET 처리:
    ├─ 각 dialogue마다 마지막 user 발화 추출
    ├─ COMET xIntent 생성
    ├─ COMET xReact 생성
    └─ 컨텍스트 = [히스토리] + [의도] + [감정]
```

---

## 📚 참고: 다른 Inputter와의 비교

### vanilla.py (그래프 없는 버전)
```python
# vanilla.py에서는?
self.valid_dataloader = DynamicBatchingLoader  # JSON Lines 직접 처리
```
- Valid도 런타임 처리 (JSON Lines 읽음)
- COMET이 없으므로 빠름

### vanilla_in_re.py (COMET 포함)
```python
# vanilla_in_re.py
self.valid_dataloader = DynamicBatchingLoader  # JSON Lines 직접 처리
```
- **하지만 DynamicBatchingLoader 내부에서 convert_data_to_inputs 호출**
- convert_data_to_inputs는 COMET 실행 (L107-127)
- **런타임에 COMET 반복 실행됨!**

---

## 🔑 핵심 교훈

```
vanilla vs vanilla_in_re 차이:

vanilla.py:
  - convert_data_to_inputs: 단순 토큰화만 (빠름)
  - 런타임 처리 가능
  
vanilla_in_re.py:
  - convert_data_to_inputs: COMET 실행 포함 (느림)
  - 훈련 중 매 epoch마다 COMET 재실행하면 매우 비효율적!
  - → 미리 prepare.py로 처리 권장
```

---

## ✨ 최종 결론

**你的直觉是对的!** (당신의 직감이 맞습니다!)

1. ✅ **후자가 맞습니다**: COMET은 사전에 처리되어야 함
2. ✅ **Train, Valid, Test 모두 처리해야 합니다**
3. ⚠️ **현재 설정이 부정확합니다**: Valid/Test는 런타임에 COMET 반복 실행 중

