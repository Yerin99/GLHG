# Blenderbot + COMET (vanilla_in_re) 사용 가이드

본 가이드는 GLHG의 Hierarchical Graph Reasoner 없이 **Blenderbot + COMET만을 사용**하여 훈련 및 추론하는 방법을 설명합니다.

## 📋 개요

**vanilla_in_re** inputter는 다음 기능을 제공합니다:
- ✅ COMET-distill을 사용하여 **xIntent**(의도)와 **xReact**(반응) 자동 추출
- ✅ Blenderbot을 기반으로 한 대화 생성
- ✅ GLHG 논문의 4.2 Experimental Setting 기준 설정

## 🔧 사전 준비

### 1. 필요한 사전 훈련 모델

#### Blenderbot Small 모델 다운로드
```bash
cd Blenderbot_small-90M
wget -O pytorch_model.bin https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/pytorch_model.bin
cd ..
```

#### COMET-distill 모델 다운로드

**참고**: [COMET-ATOMIC 2020 (GitHub)](https://github.com/allenai/comet-atomic-2020)에서 제공하는 모델입니다.

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

print("✓ COMET models downloaded successfully!")
EOF
```

**다운로드 위치 확인**:
```bash
ls -lh /data/pretrained_models/comet-distill-tokenizer/
ls -lh /data/pretrained_models/comet-distill-high/
```

> ⚠️ **주의**: 코드에서 모델 경로가 다음과 같이 하드코딩되어 있습니다 (`inputters/vanilla_in_re.py` 17-19번 라인)
> ```python
> tokenizer_gpt = GPT2Tokenizer.from_pretrained('/data/pretrained_models/comet-distill-tokenizer')
> model_gpt = GPT2LMHeadModel.from_pretrained('/data/pretrained_models/comet-distill-high').cuda()
> ```
> 경로가 다르면 수정하세요.

### 2. 데이터 형식
데이터는 JSON Lines 형식이어야 하며, 각 라인은 다음 구조를 따라야 합니다:

```json
{
  "dialog": [
    {"speaker": "usr", "text": "사용자 발화"},
    {"speaker": "sys", "text": "시스템 응답"},
    ...
  ]
}
```

## 📋 빠른 시작 가이드

**전체 실행 명령어** (가정: ESConv 데이터가 이미 _reformat/train.txt 형태로 준비됨):

```bash
# 1. 데이터 전처리 (COMET으로 의도/감정 추출)
./scripts/prepare_vanilla_in_re.sh

# 2. 모델 훈련 (5 epoch, 약 2-4시간 V100 기준)
./scripts/train_vanilla_in_re.sh

# 3. 모델 추론 및 평가
./scripts/infer_vanilla_in_re.sh
```

---

### Step 0: ESConv 데이터 전처리 (선택사항)

본 저장소의 `_reformat/` 폴더에는 ESConv 원본 데이터셋과 처리 스크립트가 포함되어 있습니다.

**파일 구조**:
```
_reformat/
├── ESConv.json          # 원본 ESConv 데이터셋 (8.6MB)
├── process.py           # 데이터 처리 스크립트
└── [생성될 파일들]
    ├── train.txt        # 훈련 데이터 (70%)
    ├── valid.txt        # 검증 데이터 (15%)
    ├── test.txt         # 테스트 데이터 (15%)
    └── sample.json      # 샘플 10개
```

**처리 방법**:
```bash
# 1. _reformat 디렉토리로 이동
cd _reformat

# 2. process.py 실행 (ESConv.json을 처리하여 train/valid/test 생성)
python process.py
# 소요 시간: 약 1-2분
```

**process.py 수행 작업**:
1. `ESConv.json`에서 대화 데이터 로드
2. 각 발화를 정규화 (normalize)
3. speaker를 'seeker' → 'usr', 'supporter' → 'sys'로 변환
4. 무작위 섞기 후 70/15/15 비율로 split
5. JSON Lines 형식으로 저장

**생성된 데이터 형식** (JSON Lines - 각 라인이 하나의 JSON 객체):
```json
{
  "emotion_type": "guilt",
  "problem_type": "debt",
  "situation": "I have more debts that...",
  "dialog": [
    {"text": "I feel so guilty about my debt situation...", "speaker": "usr"},
    {"text": "It's understandable to feel this way...", "speaker": "sys", "strategy": ["exploration"]},
    {"text": "How can I overcome this?", "speaker": "usr"},
    {"text": "Here are some steps...", "speaker": "sys", "strategy": ["information"]}
  ]
}
```

> ✅ **팁**: vanilla_in_re inputter는 이 구조의 데이터만 처리할 수 있습니다. 다른 데이터셋을 사용하려면 같은 형식으로 변환해야 합니다.

### Step 1: 데이터 전처리 (vanilla_in_re inputter - COMET 사전 처리)

**중요**: vanilla_in_re은 COMET을 사용하므로, **train, valid, test를 모두 미리 pickle로 처리**해야 합니다.
만약 JSON Lines로 남겨두면 훈련 중 매 epoch마다 COMET이 반복 실행되어 매우 느립니다!

먼저 스크립트 파일에 실행 권한을 부여합니다:
```bash
chmod +x scripts/*.sh
```

전처리 스크립트 실행:
```bash
./scripts/prepare_vanilla_in_re.sh
```

**이 스크립트가 자동으로 수행하는 작업:**

1. **Train 데이터 처리**
   - 입력: `_reformat/train.txt` (JSON Lines)
   - COMET 실행: 각 dialogue마다 xIntent, xReact 추출
   - 출력: `DATA/vanilla_in_re.vanilla_in_re/data.pkl`

2. **Valid 데이터 처리** (새로 추가!)
   - 입력: `_reformat/valid.txt` (JSON Lines)
   - COMET 실행: 각 dialogue마다 xIntent, xReact 추출
   - 출력: `DATA/vanilla_in_re_valid.vanilla_in_re/data.pkl`

3. **Test 데이터 처리** (새로 추가!)
   - 입력: `_reformat/test.txt` (JSON Lines)
   - COMET 실행: 각 dialogue마다 xIntent, xReact 추출
   - 출력: `DATA/vanilla_in_re_test.vanilla_in_re/data.pkl`

**각 처리 단계마다 동일하게 수행:**
- JSON Lines 파일의 각 line을 읽음
- dialogue에서 마지막 user 발화 추출
- COMET xIntent 생성 (사용자의 의도)
- COMET xReact 생성 (감정 반응)
- 컨텍스트 구성: [대화히스토리] + [의도] + [감정반응]
- InputFeatures로 변환하여 pickle 저장

**주요 설정**:
- `MAX_INPUT_LENGTH`: 128
- `MAX_DECODER_INPUT_LENGTH`: 50

### Step 2: 모델 훈련

훈련 스크립트 실행:

```bash
./scripts/train_vanilla_in_re.sh
```

**이 단계에서 사용되는 데이터:**
- **Train**: `DATA/vanilla_in_re.vanilla_in_re/data.pkl` (미리 처리됨) ✅
- **Valid**: `DATA/vanilla_in_re_valid.vanilla_in_re/data.pkl` (미리 처리됨) ✅

pickle 파일을 사용하므로 훈련 중에 COMET이 실행되지 않아 빠릅니다!

**주요 하이퍼파라미터 (GLHG 논문 4.2 기준)**:
- Learning Rate: `3e-5`
- Warmup Steps: `100`
- Batch Size: `16`
- Epochs: `5`
- Max Input Length: `128`
- Optimizer: AdamW (β1=0.9, β2=0.99)

훈련 결과:
- 체크포인트: `./DATA/vanilla_in_re.vanilla_in_re/{TIMESTAMP}/epoch-{N}.bin`
- 로그: `train_log.csv`, `eval_log.csv`

### Step 3: 추론

추론 스크립트에서 **체크포인트 경로를 수정**한 후 실행합니다:

```bash
# scripts/infer_vanilla_in_re.sh 파일 내 CHECKPOINT_PATH 수정
# 예: CHECKPOINT_PATH="./DATA/vanilla_in_re.vanilla_in_re/{TIMESTAMP}/epoch-4.bin"
vi scripts/infer_vanilla_in_re.sh

# 추론 실행
./scripts/infer_vanilla_in_re.sh
```

**이 단계에서 사용되는 데이터:**
- **Test**: `DATA/vanilla_in_re_test.vanilla_in_re/data.pkl` (미리 처리됨) ✅

pickle 파일을 사용하므로 추론 중에 COMET이 실행되지 않아 빠릅니다!

**주요 설정 (논문 기준)**:
- Batch Size: `1`
- Max Decoding Steps: `40`

추론 결과는 체크포인트 디렉토리 내 `res_*` 폴더에 저장됩니다:
- `gen.json`: 생성 결과 (JSON 형식)
- `gen.txt`: 생성 결과 (텍스트 형식)
- `metric.json`: 평가 메트릭

## 📊 COMET 활용 방식

`vanilla_in_re` inputter는 다음과 같이 COMET을 활용합니다:

### 전체 데이터 처리 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 0: ESConv 원본 데이터 처리                                  │
├─────────────────────────────────────────────────────────────────┤
│ ESConv.json (원본)                                              │
│     ↓                                                            │
│ _reformat/process.py (speaker 정규화, 70/15/15 split)           │
│     ↓                                                            │
│ train.txt, valid.txt, test.txt (JSON Lines 형식)               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: vanilla_in_re Inputter로 COMET 처리                     │
├─────────────────────────────────────────────────────────────────┤
│ prepare.py --inputter_name vanilla_in_re                       │
│     ↓                                                            │
│ 각 대화 샘플마다:                                               │
│   1. 사용자 마지막 발화 추출                                    │
│   2. COMET xIntent 생성 (사용자 의도 추출)                      │
│   3. COMET xReact 생성 (감정 반응 추출)                         │
│   4. 컨텍스트 재구성: [히스토리] + [의도] + [반응]              │
│   5. InputFeatures 생성                                          │
│     ↓                                                            │
│ DATA/vanilla_in_re.vanilla_in_re/data.pkl                      │
│ (COMET으로 강화된 feature 데이터)                               │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2-3: 훈련 및 추론                                          │
├─────────────────────────────────────────────────────────────────┤
│ train.py → Blenderbot 훈련                                      │
│ infer.py → 응답 생성 및 평가                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 처리 과정 (`inputters/vanilla_in_re.py` - 코드 분석)
1. **입력**: 사용자의 마지막 발화 (`last_sentence_ids`)
2. **COMET xIntent 추출**: 
   - 템플릿: `"<head> {발화} </head> <relation> xIntent </relation> [GEN]"`
   - 출력: 사용자의 의도 (예: "to get help.")
3. **COMET xReact 추출**:
   - 템플릿: `"<head> {발화} </head> <relation> xReact </relation> [GEN]"`
   - 출력: 사용자의 감정 반응 (예: "sad.")
4. **컨텍스트 구성**: `[대화 히스토리] + [xIntent] + [xReact]`
5. **모델 입력**: 구성된 컨텍스트를 Blenderbot에 입력하여 응답 생성

### 코드 위치 (`inputters/vanilla_in_re.py`):
- 모델 및 토크나이저 로드: 라인 17-21
- 데이터 변환 (`convert_data_to_inputs`): 라인 92-151
- COMET xIntent 추출: 라인 107-115
- COMET xReact 추출: 라인 118-127
- 컨텍스트 재구성: 라인 116, 128-135

## 🔍 주요 파일 설명

| 파일 | 설명 |
|------|------|
| `_reformat/ESConv.json` | 원본 ESConv 데이터셋 |
| `_reformat/process.py` | ESConv JSON → train/valid/test JSON Lines 변환 |
| `inputters/__init__.py` | vanilla_in_re inputter 등록 |
| `inputters/vanilla_in_re.py` | COMET 통합 inputter (xIntent/xReact 추출) |
| `CONFIG/vanilla_in_re.json` | Blenderbot Small 모델 설정 |
| `scripts/prepare_vanilla_in_re.sh` | 데이터 전처리 실행 스크립트 |
| `scripts/train_vanilla_in_re.sh` | 모델 훈련 스크립트 |
| `scripts/infer_vanilla_in_re.sh` | 모델 추론 스크립트 |
| `prepare.py` | 핵심: JSON Lines → pickle (feature) 변환 |
| `train.py` | 훈련 루프 |
| `infer.py` | 추론 루프 |

## 🎯 GLHG 논문 대비 차이점

본 설정은 GLHG 논문에서 다음 구성요소를 **제외**합니다:
- ❌ Hierarchical Graph Reasoner
- ❌ Global-to-Local Graph Network
- ❌ Strategy modeling

본 설정에서 **유지**하는 구성요소:
- ✅ Blenderbot as backbone
- ✅ COMET for intention extraction (xIntent)
- ✅ COMET for reaction extraction (xReact)
- ✅ 논문의 하이퍼파라미터 설정

## 💡 팁 및 문제 해결

### 1. CUDA Out of Memory
- `train_vanilla_in_re.sh`에서 `GRADIENT_ACC_STEPS`를 증가시키고 `TRAIN_BATCH_SIZE`를 감소시킵니다.
- FP16을 활성화합니다: `--fp16 True`

### 2. COMET 모델 경로 오류
- `inputters/vanilla_in_re.py` 17-19번 라인의 경로를 확인하세요:
```python
tokenizer_gpt = GPT2Tokenizer.from_pretrained('/data/pretrained_models/comet-distill-tokenizer')
model_gpt = GPT2LMHeadModel.from_pretrained('/data/pretrained_models/comet-distill-high').cuda()
```
- COMET 모델은 **전처리 시점(prepare.py)**에서만 로드되며, 훈련/추론 시점에는 필요하지 않습니다.
- 모델 경로가 없으면 `prepare.py` 실행 단계에서만 실패합니다.

### 3. 전처리가 너무 느림
- `prepare_vanilla_in_re.sh`에서 `--single_processing` 플래그를 제거하여 멀티프로세싱을 활성화합니다.
- 단, COMET이 GPU를 사용하므로 멀티프로세싱 시 GPU 메모리 부족이 발생할 수 있습니다.

### 4. 데이터 로딩 시 에러
- 데이터 형식이 올바른지 확인합니다 (JSON Lines 형식).
- `speaker` 필드가 'usr', 'sys'로 정확히 표기되었는지 확인합니다.

## 📚 참고 자료

- GLHG 논문: "Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation" (IJCAI 2022)
- COMET: [https://github.com/allenai/comet-atomic-2020](https://github.com/allenai/comet-atomic-2020)
- Blenderbot: [https://huggingface.co/facebook/blenderbot-400M-distill](https://huggingface.co/facebook/blenderbot-400M-distill)

## 🤔 다음 단계

1. **베이스라인 성능 확인**: vanilla_in_re로 훈련/추론 후 성능 측정
2. **COMET 효과 분석**: xIntent와 xReact 추출이 성능에 미치는 영향 분석
3. **GLHG와 비교**: 본 설정과 전체 GLHG 구현의 성능 차이 비교

---

**작성일**: 2026-01-19  
**작성자**: AI Research Assistant
