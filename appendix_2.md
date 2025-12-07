# Appendix – Experimental Setup (Phase 2: RL 기반 n-word 요약)

본 Phase에서는 문장 선택(sentence selection)과 청크 선택(chunk selection)의 두 단계에  
강화학습(Reinforcement Learning)을 적용하여 n-word compressive summarization의 품질을 향상시키는 실험을 수행하였다.

---

## 1. 하드웨어 환경

- 실행 환경: **Google Colab (무료 버전)**
- GPU: **NVIDIA T4 (16GB VRAM)**
- 시스템 메모리(RAM): 약 **12–16GB**

모든 실험은 단일 GPU에서 실행하였으며,  
분산 학습이나 mixed precision은 사용하지 않았다.

---

## 2. 소프트웨어 / 라이브러리 버전

- Python: Colab 기본 Python 3.x  
- PyTorch: Colab 기본 `torch` (CUDA 지원)  
- Transformers: `transformers`  
- Datasets: `datasets`  
- Sentence-Transformers: `sentence-transformers`  
- spaCy: `spacy`, 영어 모델 `en_core_web_sm`  
- evaluate: BERTScore 및 ROUGE 계산  
- 기타 유틸: numpy, pandas, tqdm 등

### 사용한 주요 프리트레인 모델
- **Sentence-BERT**: `all-MiniLM-L6-v2`  
  (문서/문장/청크 embedding)
- **BERT-base-uncased**  
  (BERTScore 계산용)

---

## 3. 데이터셋 설정

- 데이터셋: `cnn_dailymail` (v3.0.0)
- 사용 split  
  - `train` 일부 → **RL 학습용 (NUM_TRAIN = 200)**  
  - `test` 일부 → **최종 평가용 (NUM_TEST = 200)**  
- 각 샘플 구성  
  - `article`: 뉴스 기사 본문  
  - `highlights`: reference summary (Semantic Reward 계산용)

> Phase 2에서는 reference는 “정답 문장”으로 사용되지 않으며  
오직 semantic similarity 기반 reward 계산에만 사용됨.

---

## 4. 학습 설정: RL + 추출 파이프라인

### 4.1 요약 길이 설정  
N_LIST = [7, 9, 11, 13]
이 값들은 Phase 1에서 가장 의미 보존이 안정적으로 나타난 n 구간을 기반으로 선정하였다.

---

### 4.2 Episode / Batch 방식
- 기사(document) 1개 = 1 episode  
- batch size = 1  
- 학습 루프는 문서 단위로 순차 진행

---

## 4.3 문장 선택 및 정책 모델(Sentence Policy)

### (1) 문장 임베딩
- SBERT(`all-MiniLM-L6-v2`)로 문장 embedding  
- cosine similarity 기반 adjacency matrix 생성 → TextRank 적용

### (2) TextRank 초기 중요도
각 문장에 대해 TextRank 점수 \( t_i \) 산출  
→ 이는 “그래프 기반 구조적 중요도”를 의미

### (3) SentencePolicyModel 입력
각 문장에 대해 다음 두 정보를 concat한 feature 사용:

\[
x_i = [\,s_i \;\oplus\; t_i\,]
\]

- \( s_i \): SBERT 문장 임베딩 (semantic)
- \( t_i \): TextRank 점수 (structural importance)

### (4) 정책 모델 구조
- 3-layer MLP  
- tanh 활성화 포함  
- 출력: 문장별 logit \( z_i \)

### (5) 문장 선택
- softmax(logits) 기반 확률 분포  
- 확률이 높은 상위 K 문장 선택  
- 본 실험에서 K = 3 (semantic coverage와 RL 안정성의 균형)

---

## 4.4 청크 선택 및 정책 모델(Chunk Policy)

- spaCy noun_chunk, verb, adj 기반 의미 단위 추출  
- SBERT embedding으로 chunk-level representation 생성  
- ChunkPolicyModel이 각 chunk에 대한 delta score 출력  
- 최종 word-level score로 확산 후 best span 추출

---

## 4.5 RL Reward 설계

- 최종 생성된 N-word 요약(pred)과 reference 요약(ref) 간의  
  **BERTScore(F1)** 을 reward로 사용:

\[
R = BERTScore(pred, ref)
\]

- REINFORCE 기반 policy gradient 적용  
- reward는 문장 정책 + 청크 정책 모두에 전달되어  
  문장 선택 및 의미 단위 선택을 동시에 개선

> BERTScore 단일 reward는 semantic bias를 유도하지만  
단어 생성이 아닌 “선택 기반 요약” 문제에서는  
의미 충실도 향상에 긍정적으로 작용함.

---

## 5. 재현성 관련

- HuggingFace Hub에서 데이터 직접 스트리밍  
- 실험 전 random seed 고정  
- 동일한 장비/라이브러리 환경에서 재현 가능  
- 정책 모델 가중치는 `.pt` 파일로 저장 가능 (chunk_policy.pt, sentence_policy.pt)

---

# Appendix – Phase 2 Metrics

Phase 2의 주요 목적은  
**RL이 sentenceBERT baseline 대비 어떤 방향으로 요약 품질을 변화시켰는지 분석하는 것**이며,  
이를 위해 Efficiency와 Density 두 지표를 도입하였다.

---

## 1. Efficiency (효율성)

### 정의
단어 1개가 의미적으로 얼마나 기여했는지를 평가하는 지표:

\[
Efficiency = \frac{BERTScore(pred, ref)}{\text{요약 단어 수}}
\]

### 의도
- 같은 길이라도 더 compact한 요약인지 판단
- 의미적 정보량 대비 단어 수 효율을 측정

### 해석
- 높을수록 **단어 하나당 의미 기여도가 높음**
- semantic precision per word에 해당

---

## 2. Density (단어 재사용률)

### 정의
요약에 포함된 단어 중 원문에도 등장하는 비율:

\[
Density = \frac{|Pred \cap Original|}{|Pred|}
\]

### 의도
- 원문 핵심 표현을 얼마나 재사용했는지 평가
- extractive 성향 분석에 유용

### 해석
- 높을수록 원문과의 정보 교집합 크기가 큼
- compressive summarization에서 “핵심 정보 보존률”처럼 해석 가능

---

## 3. Efficiency & Density Trade-off

- **Efficiency ↑ → Density ↓**  
  fewer-but-stronger 단어 선택 → coverage 감소
- **Density ↑ → Efficiency ↓**  
  coverage 증가 → 단어당 의미 효율 감소

두 지표는 상호보완적이며  
Efficiency는 **압축성(compactness)**,  
Density는 **coverage(원문 정보 보존)** 을 나타냄.

---

## 4. 코드 스니펫

```python
def compute_density(pred, original):
    eff = bertscore_f1 / max(1, len(pred.split()))


def compute_density(pred, original):
    pred_tokens = simple_tokenize(pred)
    orig_tokens = simple_tokenize(original)

    pred_set = set(pred_tokens)
    orig_set = set(orig_tokens)

    if not pred_tokens:
        return 0.0

    overlap = pred_set & orig_set
    return len(overlap) / len(pred_tokens)





# Appendix – Limitations & Remaining Challenges (Phase 2)

본 절에서는 RL 기반 n-word 요약 실험에서 확인된 기술적·구조적 한계와  
향후 개선 가능성을 정리한다.

---

## 1. 강화학습 데이터 규모의 한계

### 문제점
- RL 학습에 사용된 데이터는 **CNN/DailyMail train split 중 200개 기사**로 제한되었다.
- 문장 선택은 high-variance action space를 가지므로  
  **적은 episode 수는 policy의 안정적 수렴을 방해**할 가능성이 있다.

### 영향
- SentencePolicyModel이 특정 스타일에 과적합(overfit)될 위험
- 문장 길이·문장 구조에 민감해지는 현상 발생 가능

### 개선 방향
- 더 큰 규모(수천~수만)의 RL episode 사용  
- policy warm-up (e.g., imitation learning, ranking loss) 도입  
- PPO 등 안정화 기법 적용

---

## 2. BERTScore 단일 Reward의 Bias

### 문제점
- Reward가 **오직 BERTScore(F1)** 로만 구성됨  
- 이는 “semantic similarity ↑” 방향으로만 강화되기 때문에  
  **원문 단어 재사용이 과도하게 촉진되는 bias**가 생길 수 있음  
  → Density가 과도하게 증가할 여지가 있음

### 영향
- Coverage는 잘 잡지만  
  Efficiency(단어당 의미량)는 baseline 대비 상대적으로 낮아질 수 있음  
- 모델이 “reference 문체”에 종속되는 경향도 존재

### 개선 방향
- Multi-reward 구조 도입:
  - BERTScore + ROUGE-L + Length penalty  
- Diversity 보상 추가  
- Reference-less reward (e.g., QAEval, BLEURT) 고려

---


## 3. TextRank 기반 초기 스코어의 구조적 한계

### 문제점
- TextRank는 문장 간 유사성 기반 그래프에서 중요도를 계산하지만
  - 문서 구조를 반영하지 못함  
  - 사건 흐름이나 causal relation을 고려하지 못함
- RL이 이를 보정하도록 설계되어 있으나  
  **초기 스코어가 불안정하면 policy도 불안정하게 시작**됨

### 영향
- SentencePolicyModel이 탐색해야 하는 space가 넓어짐  
- 초기 문장 선택 품질이 크게 variance를 만듦

### 개선 방향
- Discourse parser 기반 문장 연결성 신호 추가  
- SBERT 대신 LongFormer/BigBird 임베딩 실험  
- TextRank replacement: LexRank, PacSum 등 비교

---

## 4. Fixed Top-K Sentence Selection (K=3)

### 문제점
- 모든 기사에서 “중요 문장은 3개일 것”이라는 가정  
- 실제로는 문서의 길이·구조에 따라  
  중요한 문장은 1개 또는 4개 이상일 수 있음

### 영향
- 짧은 문서는 지나치게 많은 문장이 선택됨  
- 긴 문서는 중요한 문장이 누락될 가능성

### 개선 방향
- Dynamic K selection:  
  - TextRank 스코어 기반 thresholding  
  - RL이 직접 K를 선택하도록 하는 hierarchical policy  
- Coverage-oriented stopping policy 도입

---

## Summary

본 Phase 2는 RL을 활용해 n-word extractive–compressive summarization을 구현하는 과정에서  
다양한 구조적·데이터적·지표적 제약을 마주하였다.

그러나 이러한 한계는  
향후 Multi-Reward RL, Discourse-aware 문장 선택,  
Dynamic K Policy 등으로 보완 가능하며  
본 연구의 발전 방향을 제시하는 중요한 근거가 된다.