# Appendix – Experimental Setup (Phase 1: Magic Number *n* 찾기)

Phase 1에서는 각 기사에서 의미를 가장 잘 보존하는 단어 수 *n*을 찾기 위한 탐색 실험을 수행하였다.  
다양한 프리트레인 모델(BERT, XLNet, ELECTRA, BART)을 활용하여 문장 중요도를 계산하고,  
여러 n-word 요약을 생성한 뒤 BERTScore와 ROUGE로 평가하였다.

---

## 1. 하드웨어 환경

- 실행 환경: **Google Colab (무료 버전)**
- GPU: **NVIDIA T4 (16GB)**
- 시스템 메모리: **12–16GB**

모든 실험은 단일 GPU(T4)에서 수행되었으며 멀티 GPU 분산 학습은 사용하지 않았다.

---

## 2. 소프트웨어 / 라이브러리 버전

- Python: Colab 기본 Python 3.x  
- PyTorch: Colab 기본 torch (CUDA 지원)  
- Transformers: HuggingFace `transformers`  
- Datasets: HuggingFace `datasets`  
- evaluate: BERTScore, ROUGE 계산  
- 기타: numpy, pandas, tqdm, spaCy 등

사용된 주요 프리트레인 모델은 다음과 같다:

### Encoder 모델 (문장/청크 임베딩, 중요도 계산)
- **BERT-base-uncased**
- **XLNet-base-cased**
- **ELECTRA-base-discriminator**

### Decoder 모델 (XSUM 전용 요약 baseline)
- **BART-large-xsum**

이들 모델은 Phase 1에서 다양한 n-word 요약 후보를 생성하기 위한 baseline 역할을 수행했다.

---

## 3. 데이터셋 설정

Phase 1에서 사용한 데이터셋은 다음 두 가지이다:

### ✓ CNN/DailyMail (3.0.0)
- 사용 split: `test`
- 사용 샘플 수: **약 2,000개**
- 입력: `article`
- reference: `highlights`

### ✓ XSUM
- 사용 split: `test`
- 사용 샘플 수: **약 2,000개**
- 입력: `document`
- reference: `summary`
- BART-XSum baseline과 성능 비교에 활용

reference summary는 Phase 1에서 학습이 아닌 **평가지표 계산용**으로만 사용되었다.

---

## 4. 실험 설정 (Magic Number *n* 탐색)

Phase 1의 목표는 **각 문서에 대해 가장 의미 보존도가 높은 요약 길이 n을 찾는 것**이다.

### 사용된 n 값
N_LIST = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]

각 문서마다 위 모든 n 값으로 요약을 생성한 뒤  
BERTScore 및 ROUGE-1/ROUGE-2 값을 측정하여  
길이가 의미 보존에 미치는 영향을 비교하였다.

### 실행 방식
- batch가 아닌 **문서 단위 sequential evaluation**
- 실질 batch size = **1**

### 역할 분담
- BERT/XLNet/ELECTRA → 문장 임베딩 및 의미 중요도 평가
- BART-XSum → XSUM에서 baseline 생성 요약으로 비교

---

## 5. 평가 지표

Phase 1에서는 다음 두 지표를 사용하였다:

### BERTScore
- embedding 기반 semantic similarity 평가  
- n-word compressive summarization에서도 안정적 평가 가능

### ROUGE-1 / ROUGE-2
- reference summary와의 unigram 및 bigram 중복률  
- 정보 보존 경향을 직관적으로 파악하기 위한 보조 지표

이 두 지표를 통해 길이가 증가함에 따라 의미 손실, 의미 보존, 중복 패턴을 분석하였으며,  
그 결과 **Phase 2에서 사용할 n 값 구간(7, 9, 11, 13)**을 선정하였다.

---

## 6. 재현성 관련

- 데이터는 HuggingFace Hub에서 직접 다운로드  
- 모든 실험은 일관된 환경에서 실행  
- Phase 1 결과는 Phase 2의 RL 정책 설계에 활용됨

---


# Appendix - Phase 1 Mertics

본 문서는 본 연구에서 magic word를 탐색하기 위한 
두 가지 핵심 메트릭 **ROUGE** 및 **BBERTScore**의 정의와 해석을 정리합니다.

## 1. ROUGE

### 정의
ROUGE(Recall-Oriented Understudy for Gisting Evaluation)는  
**reference summary와 생성 요약 간의 n-gram 중복률**을 측정하는 대표적인 추출 요약 평가 지표입니다.

본 연구에서는 다음 두 가지를 사용했습니다:

- **ROUGE-1:** unigram(단어 단위) 중복 비율  
- **ROUGE-2:** bigram(연속된 두 단어) 중복 비율  

ROUGE는 다음과 같이 정의됩니다:

\[
\text{ROUGE-n} = 
\frac{\text{reference와 prediction 사이의 n-gram 교집합 수}}
{\text{reference의 전체 n-gram 수}}
\]

즉, reference에 있는 정보가 모델 요약에 얼마나 재현되었는지를 보상하는 지표입니다.

---

### 의도
- **요약이 reference에서 나타난 핵심 단어·연결 표현을 얼마나 보존했는지 평가**
- 특히 **추출 기반(extractive)** 또는 **압축 기반(compressive)** 요약에서
  정보 보존 수준을 파악하기 위해 사용

---

### 해석
- **높을수록 reference 요약에 더 충실**  
- ROUGE-2가 낮으면 중요한 문맥이나 연결 관계를 적절히 재현하지 못했을 가능성이 큼  
- ROUGE-1은 전체 핵심 단어 보존량을,  
  ROUGE-2는 핵심 단어들의 “연속적인 맥락” 보존을 측정함

---

### 예시
reference:  
> "the international court opened investigation"

prediction:  
> "court opened investigation"

- **ROUGE-1:**  
  reference 단어 4개 중 3개가 prediction에 포함 → 3/4 = 0.75

- **ROUGE-2:**  
  reference bigram 3개 중 “opened investigation” 1개만 일치 → 1/3 ≈ 0.33

---



## 2. BERTScore

### 정의
BERTScore는 reference summary와 생성 요약(prediction) 간의  
**의미적 유사도(semantic similarity)** 를 평가하는 지표입니다.  

기존 ROUGE처럼 n-gram 중복을 세는 방식이 아니라,  
**BERT 기반 contextual embedding**을 사용해  
각 단어(token) 간 cosine similarity를 계산한 뒤  
precision, recall, F1 형태로 요약 전체의 의미 유사도를 측정합니다.

대표적인 계산 개념은 다음과 같습니다:

- reference의 각 token과 가장 유사한 prediction token의 cosine similarity를 찾고  
- 이를 평균내어 전체 의미 유사도를 산출  
- 최종적으로 F1 형태가 가장 안정적이기 때문에 F1 점수를 사용

---

### 의도
- 단순 단어 중복이 아니라  
  **동의어, 문맥적 의미, paraphrasing까지 반영하여 실제 의미 보존도를 평가**
- 추출/압축 기반 요약에서  
  “reference가 표현하는 의미를 얼마나 유지했는가”를 정밀하게 평가하기 위함
- Compressive summarization에서는 ROUGE보다 기준이 덜 경직적이고  
  semantic-level 평가가 가능해 reward로도 활용하기 적합함

---

### 해석
- **높을수록 reference와 의미적으로 더 가까운 요약**  
- ROUGE가 단어 매칭 기반의 *형태적 유사도*라면  
  BERTScore는 embedding 기반의 *의미적 유사도*  
- 특히 단어 순서가 달라지거나 표현이 바뀌어도  
  전체 의미가 같으면 높은 점수를 받을 수 있음  
- RL reward로 사용 시, semantic fidelity(의미적 충실도)를 강화하는 방향으로 학습됨

---

### 예시
reference:
> “the court opened an investigation”

prediction:
> “investigation was launched by the court”

ROUGE-1은 낮게 나올 수 있지만  
BERTScore는 두 문장이 **의미적으로 거의 동일**하기 때문에  
약 0.90 이상의 높은 점수를 얻을 수 있다.

---


## Appendix – Experimental Setup (Phase 2: RL 기반 n-word 요약)

### 1. 하드웨어 환경

- 실행 환경: Google Colab (무료 버전)
- GPU: NVIDIA T4 1기
  - GPU 메모리: 16 GB
- 시스템 메모리(RAM): 약 12–16 GB 범위 (Colab 기본 제공 스펙)

본 실험은 모두 단일 GPU(T4)에서 수행되었으며, 멀티 GPU 분산 학습은 사용하지 않았다.

---

### 2. 소프트웨어 / 라이브러리 버전

- Python: Colab 기본 Python 3.x
- PyTorch: Colab 기본 `torch` 버전 (CUDA 지원)
- Transformers: `transformers` (HuggingFace)
- Datasets: `datasets` (HuggingFace)
- Sentence-Transformers: `sentence-transformers`
- spaCy: `spacy`, 영어 모델 `en_core_web_sm`
- 기타:
  - `evaluate` (BERTScore, ROUGE 계산)
  - `tqdm`, `numpy`, `pandas` 등 기본 유틸

사용한 주요 프리트레인 모델은 다음과 같다.

- Sentence-level embedding:  
  - `all-MiniLM-L6-v2` (Sentence-BERT)
- Token/semantic similarity 기반 평가지표:  
  - `bert-base-uncased` (BERTScore 계산에 사용)

---

### 3. 데이터셋 설정

- 데이터셋: HuggingFace `cnn_dailymail`, 버전 `"3.0.0"`
- 사용 split:
  - `train` split 일부 → RL 학습용
  - `test` split 일부 → 최종 평가용
- 샘플 수:
  - `NUM_TRAIN = 200`  (RL 학습에 사용한 기사 수)
  - `NUM_TEST  = 200`  (최종 평가에 사용한 기사 수)

각 기사에 대해:
- `article`: 원문 뉴스 본문
- `highlights`: reference summary로 사용  
RL에서는 이 reference summary와의 BERTScore를 reward로 사용하였다.

---

### 4. 학습 설정 (RL / 추출 파이프라인)

- 요약 길이 설정:  
  - `N_LIST = [7, 9, 11, 13]`  
  각 기사에 대해 7, 9, 11, 13 단어 길이의 n-word 요약을 생성하고 평가.
- Episode/Batch 방식:
  - 배치 학습이 아니라 **기사 단위 에피소드**로 순차 처리
  - 코드 구조상 한 번에 하나의 기사만 처리하므로,  
    **실질적인 batch size는 1에 해당**
- 문장/청크 선택:
  - 문장 임베딩: Sentence-BERT (`all-MiniLM-L6-v2`)
  - 문장 중요도 초기값: TextRank (SBERT cosine similarity 기반 그래프)
  - 문장 정책(SentencePolicyModel):  
    - 입력: `[SBERT 문장 임베딩 s_i ⊕ TextRank 점수 t_i]`
    - 출력: 문장별 logit → softmax로 K개 문장 선택
  - 청크 정책(ChunkPolicyModel):  
    - spaCy 기반 chunk, SBERT 임베딩을 이용해 chunk-level score 보정
- RL Reward:
  - 선택된 문장 + 청크 기반으로 N-word 요약 생성
  - `highlights`(reference)와의 **BERTScore(F1)** 을 reward로 사용
  - reward ↑ 방향으로 policy gradient를 적용해  
    sentence-level, chunk-level 정책을 업데이트

---

### 5. 재현성 관련

- 데이터셋은 HuggingFace Hub에서 직접 스트리밍/다운로드
- 실험은 Colab 세션에서 순차적으로 수행되었으며,
  각 Phase는 동일한 모델/지표 설정 하에서 비교 가능하도록 구성하였다.



# Appendix – Phase 2 Metrics

본 문서는 본 연구에서 n-word compressive summarization의 성능 평가를 위해 사용한  
두 가지 핵심 메트릭 **Efficiency** 및 **Density**의 정의와 해석을 정리합니다.

## 1. Efficiency (효율성)

### 정의
Efficiency는 요약문 *단어 1개당* 달성된 의미적 품질을 측정하는 지표입니다.  
본 연구에서는 **BERTScore(F1)** 를 기반으로 다음과 같이 계산했습니다:

`Efficiency = BERTScore(pred, reference) / 요약 단어 수`

### 의도
- 같은 길이의 n-word 요약이라도,  
  **단어 하나가 얼마나 많은 의미적 기여를 하는지** 평가하기 위함  
- “더 적은 단어로 더 많은 의미를 전달하는 요약”을 선호하기 위한 압축성 지표

### 해석
- **높을수록 좋음**  
- Efficiency는 “semantic precision per word”에 가까운 개념

### 예시
- 요약 길이: 9 단어  
- BERTScore(F1): 0.42  
- Efficiency = 0.42 / 9 = 0.046

---

## 2. Density (단어 재사용률)

### 정의
Density는 **요약문에 포함된 단어 중 원문에도 등장하는 단어의 비율**입니다:

`Density = |Pred ∩ Original| / |Pred|`

### 의도
- 원문의 핵심 표현·명사구를 얼마나 재사용했는지 측정  
- Extractive-like summarization 성향을 파악하는 데 효과적

### 해석
- **높을수록 원문의 중요 단어를 더 많이 포함**
- CNN/DailyMail 같은 뉴스에서는 핵심 단어가 원문에 그대로 존재하는 경우가 많아  
  density를 핵심 정보 보존률처럼 활용 가능

### 예시
- pred 단어 12개 중 11개가 원문에도 등장  
- Density = 11 / 12 = 0.92

---

## 3. Efficiency & Density Trade-off

### Efficiency ↑ → Density ↓
- fewer-but-stronger 단어 선택  
- coverage 감소

### Density ↑ → Efficiency ↓
- coverage 증가  
- 단어당 기여도 감소

두 지표는 서로 보완적이며,  
Efficiency는 **압축성(compactness)**, Density는 **coverage(정보 보존)** 를 나타냅니다.

---

## 4. 코드 스니펫

```python
eff = bertscore_f1 / max(1, len(pred.split()))
```

```python
def compute_density(pred, original):
    pred_tokens = simple_tokenize(pred)
    orig_tokens = simple_tokenize(original)

    pred_set = set(pred_tokens)
    orig_set = set(orig_tokens)

    if not pred_tokens:
        return 0.0

    overlap = pred_set & orig_set
    return len(overlap) / len(pred_tokens)
```

---

## 5. 결론

Efficiency는 “단어 하나가 가진 의미 효율”,  
Density는 “원문과의 정보 교집합 정도”를 나타냅니다.  

두 지표를 함께 사용함으로써  
semantic quality + information coverage 를 동시에 해석할 수 있으며  
n-word compressive summarization 성능 비교에 효과적으로 활용됩니다.
