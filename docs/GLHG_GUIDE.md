# GLHG 실행 가이드

> Control Globally, Understand Locally: A Global-to-Local Hierarchical Graph Network for Emotional Support Conversation (IJCAI 2022)
> 재현 구현 기준 — `models/glhg_blenderbot_small.py`, `models/hierarchical_graph.py`

---

## 환경 설정

```bash
conda env create -f env.yml
conda activate glhg
```

필요한 사전 준비물:
- GPU (COMET 추론 포함, 최소 11GB VRAM 권장)
- BlenderBot-Small-90M 모델 가중치: `./Blenderbot_small-90M/`
- COMET-ATOMIC-2020-BART 모델: `/home/yerin/pretrained_models/comet-bart-ai2` (mismayil/comet-bart-ai2)

---

## Step 1: 데이터 전처리 (Prepare)

ESConv 데이터를 COMET으로 처리하여 pickle로 저장합니다.
COMET이 각 발화마다 xReact / xIntent를 생성하므로 **GPU 필요**, 시간 소요.

```bash
bash scripts/prepare_glhg.sh
```

- 입력: `./_reformat/train.txt`, `valid.txt`, `test.txt`
- 출력: `./DATA/glhg.glhg/<timestamp>/data.pkl`

> 이미 전처리된 `data.pkl`이 있으면 이 단계는 건너뛸 수 있습니다.

---

## Step 2: 학습 (Train)

논문 하이퍼파라미터 그대로 사용합니다.

```bash
bash scripts/train_glhg.sh
```

| 파라미터 | 값 |
|----------|-----|
| Learning rate | 3e-5 |
| Batch size | 16 |
| Epochs | 5 |
| Warmup steps | 100 |
| λ₁ = λ₂ | 0.5 |
| Max input length | 128 |
| Max decoder length | 50 |
| Seed | 42 |

- 출력: `./DATA/glhg.glhg/<timestamp>/epoch-{0..4}.bin`
- 학습 로그: `train_glhg.log`
- Validation PPL은 `eval_log.csv` 확인

**Epoch 선택 기준**: val PPL 최저 epoch 선택 (재현 실험에서는 epoch-1이 best)

---

## Step 3: 추론 (Infer)

```bash
bash scripts/infer_glhg.sh <checkpoint_path>
```

예시:
```bash
bash scripts/infer_glhg.sh ./DATA/glhg.glhg/2026-02-28133257.3e-05.16.1gpu/epoch-1.bin
```

| 디코딩 파라미터 | 값 |
|----------------|-----|
| top_k | 0 |
| top_p | 0.9 |
| temperature | 0.7 |
| num_beams | 1 |
| no_repeat_ngram_size | 3 |
| min_length | 10 |
| max_length | 40 |

- 출력: `<checkpoint_dir>/res_epoch-*.bin_test_.../metric.json`, `gen.json`

---

## 결과 확인

```bash
cat DATA/glhg.glhg/<timestamp>/res_*/metric.json
```

**재현 실험 결과** (2026-03-01, COMET-BART-AI2, epoch-2):

| PPL↓ | B-1↑ | B-2↑ | B-4↑ | D-1↑ | D-2↑ | R-L↑ |
|------|------|------|------|------|------|------|
| 16.31 | 17.26 | 6.49 | 1.84 | 3.88 | 22.32 | 16.18 |

> 이전 결과 (2026-02-28, comet-distill-high 사용 — 비학습 모델, 결과 무효):
> PPL 16.35, B-1 17.72, B-2 6.36, B-4 1.57, D-1 3.64, D-2 21.99, R-L 15.63

**논문 보고 수치** (IJCAI 2022, Table 1):

| PPL↓ | B-1↑ | B-2↑ | B-3↑ | B-4↑ | D-1↑ | D-2↑ | R-L↑ |
|------|------|------|------|------|------|------|------|
| 15.67 | 19.66 | 7.57 | 3.74 | 2.13 | 3.50 | 21.61 | 16.37 |

---

## 주요 구현 파일

| 파일 | 역할 |
|------|------|
| `models/glhg_blenderbot_small.py` | GLHG 메인 모델 (Multi-source Encoder + Fusion Gate + Decoder) |
| `models/hierarchical_graph.py` | Hierarchical Graph Reasoner (2-layer GAT) |
| `inputters/glhg.py` | 데이터 처리 (COMET xIntent, COMET-BART-AI2 사용) |
| `CONFIG/glhg.json` | 모델 설정 |
| `scripts/prepare_glhg.sh` | 전처리 스크립트 |
| `scripts/train_glhg.sh` | 학습 스크립트 |
| `scripts/infer_glhg.sh` | 추론 스크립트 |

구현 상세 내용은 `docs/GLHG_IMPLEMENTATION.md` 참조.

---

## 주의사항

- **Problem type 수**: 논문은 12개, 코드 구현은 13개로 불일치 존재
- **GAT head 수**: 논문 본문 미기재, 코드 구현은 4-head
- **COMET 관계**: xIntent만 사용. comet-distill-high는 손상된 모델(base GPT-2, COMET 학습 없음)이므로 COMET-BART-AI2로 교체함.
- `Blenderbot_small-90M/pytorch_model.bin`은 용량이 크므로 git 관리 제외 (`.gitignore` 처리됨)
