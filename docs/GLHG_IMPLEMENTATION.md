# GLHG (Global-to-Local Hierarchical Graph Network) 구현 가이드

## 개요

이 문서는 IJCAI 2022 논문 "Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation"의 완전한 구현을 설명합니다.

### 논문 정보
- **저자**: Wei Peng, Yue Hu, Luxi Xing, Yuqiang Xie, Yajing Sun, Yunpeng Li
- **기관**: Institute of Information Engineering, Chinese Academy of Sciences
- **논문**: [IJCAI 2022]

## 아키텍처

GLHG는 세 가지 주요 컴포넌트로 구성됩니다:

### 1. Multi-Source Encoder (다중 소스 인코더)

- **Contextual Encoder (Enc_ctx)**: 대화 이력 인코딩
  - 입력: `C = ([CLS], u_1, [SEP], ..., u_{N-1}, [SEP])`
  - 출력: 토큰별 은닉 상태 `(h_1, h_2, ..., h_T)`

- **Global Encoder (Enc_glo)**: Situation 정보 인코딩
  - 수식: `g = Max-pooling(Enc_glo(s_1, ..., s_P))`
  - 출력: 전역 원인 임베딩 `g ∈ R^d`

- **Local Encoder (Enc_loc)**: COMET xIntent 인코딩
  - 수식: `l = Max-pooling(Enc_loc(ms_1, ..., ms_L))`
  - 출력: 지역 의도 임베딩 `l ∈ R^d`

### 2. Hierarchical Graph Reasoner (계층적 그래프 추론기)

**노드 구성**: `V = {g, h_1, ..., h_T, l}`

**엣지 구성**:
- **Global Connection**: 전역 노드 `g`가 모든 노드와 연결
- **Local Connection**: 지역 노드 `l`이 마지막 발화의 토큰들 및 전역 노드와 연결
- **Contextual Connection**: 토큰 노드들이 인접 토큰들과 연결 (window_size=5)

**Graph Attention (수식 4-8)**:
```
α_j = softmax(LeakyReLU(a^T [W*v_i || W*v_j]))
v_i^(k+1) = σ(Σ α_j * W * v_j^(k))
```

### 3. Global-guide Decoder (전역 가이드 디코더)

- **응답 생성**: 업데이트된 그래프 노드 `v^(K)`와 Cross-attention
- **Problem Type 분류**: `p(o) = Softmax(MLP(v_1^(K)))`

### 4. Joint Training (수식 13)

```
L(θ) = λ_1 * L_1 + λ_2 * L_2
```
- `L_1`: 응답 생성 손실 (Negative Log-Likelihood)
- `L_2`: Problem type 분류 손실 (Cross-Entropy)
- `λ_1 = λ_2 = 0.5`

## 파일 구조

```
GLHG/
├── models/
│   ├── __init__.py              # 모델 등록
│   ├── glhg_blenderbot_small.py # GLHG 메인 모델
│   └── hierarchical_graph.py    # Graph Reasoner 모듈
├── inputters/
│   ├── __init__.py              # Inputter 등록
│   └── glhg.py                  # GLHG 데이터 처리
├── CONFIG/
│   └── glhg.json                # 모델 설정
├── scripts/
│   ├── prepare_glhg.sh          # 데이터 전처리
│   ├── train_glhg.sh            # 훈련
│   └── infer_glhg.sh            # 추론
└── GLHG_IMPLEMENTATION.md       # 이 문서
```

## 사용 방법

### 1. 데이터 전처리

COMET을 사용하여 xIntent를 생성합니다 (시간 소요):

```bash
cd /home/yerin/baseline/GLHG
bash scripts/prepare_glhg.sh
```

> **COMET 모델**: COMET-BART-AI2 (`mismayil/comet-bart-ai2`) 사용.
> 경로: `/home/yerin/pretrained_models/comet-bart-ai2`
> (comet-distill-high는 학습되지 않은 GPT-2 base로 사용 불가)

### 2. 훈련

논문 설정 (learning rate: 3e-5, batch size: 16, epochs: 5):

```bash
bash scripts/train_glhg.sh
```

또는 직접 실행:

```bash
python train.py \
    --config_name glhg \
    --inputter_name glhg \
    --train_batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 5 \
    --warmup_steps 100 \
    --max_input_length 128 \
    --eval_input_file ./_reformat/valid.txt
```

### 3. 추론

```bash
bash scripts/infer_glhg.sh ./DATA/glhg.glhg/<timestamp>/epoch-4.bin
```

## 논문 기대 성능

| Metric | Value |
|--------|-------|
| PPL | 15.67 |
| BLEU-1 | 19.66 |
| D-1 | 3.50 |

### 훈련 시 주의사항

- 한 epoch 후 PPL이 25 이상이면 문제가 있을 수 있음
- 정상적인 훈련에서 PPL은 점진적으로 감소해야 함
- Classification loss도 함께 감소하는지 확인

## 주요 하이퍼파라미터

| Parameter | Value | Description |
|-----------|-------|-------------|
| learning_rate | 3e-5 | AdamW learning rate |
| batch_size | 16 | Mini-batch size |
| epochs | 5 | Training epochs |
| warmup_steps | 100 | Linear warmup steps |
| max_input_length | 128 | Maximum input sequence length |
| max_decoder_input_length | 50 | Maximum decoder input length |
| λ_1, λ_2 | 0.5, 0.5 | Loss weights |
| K (graph layers) | 2 | Number of graph reasoning layers |
| num_heads | 4 | Attention heads in GAT |
| window_size | 5 | Contextual connection window |

## Problem Types (13 classes)

ESConv 데이터셋의 13가지 problem types:

1. academic pressure
2. Alcohol Abuse
3. Appearance Anxiety
4. breakup with partner
5. conflict with parents
6. Issues with Children
7. Issues with Parents
8. job crisis
9. ongoing depression
10. problems with friends
11. Procrastination
12. School Bullying
13. Sleep Problems

## 기술적 세부사항

### Graph Attention Layer

```python
# 어텐션 스코어 계산
F(v_i, v_j) = LeakyReLU(a^T [W*v_i || W*v_j])

# 소프트맥스 정규화
α_j = exp(F(v_i, v_j)) / Σ exp(F(v_i, v_j'))

# 노드 업데이트
v_i^(k+1) = σ(Σ α_j * W * v_j^(k))
```

### Multi-Source Fusion

그래프 업데이트된 표현과 원본 인코더 출력을 게이팅 메커니즘으로 융합:

```python
gate = sigmoid(W_g * [encoder_output || graph_output])
fused = gate * graph_output + (1 - gate) * encoder_output
```

## 참고 문헌

```bibtex
@inproceedings{peng2022glhg,
  title={Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation},
  author={Peng, Wei and Hu, Yue and Xing, Luxi and Xie, Yuqiang and Sun, Yajing and Li, Yunpeng},
  booktitle={IJCAI},
  year={2022}
}
```
