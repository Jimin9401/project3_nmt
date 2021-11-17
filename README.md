## Baseline model of Goorm Project-3 NMT

영-> 한 번역을 위한 Encoder-Decoder 모델

- 한국어 번역을 위해 Decoder에서 `monologg/koelectra-base-v3-discriminator`의 `token_embeddings`을 한국어 pretrained Subword Embedding으로 사용합니다.
- 영어 데이터를 위해 Encoder에서 `bert-base-uncased`의 pretrained model로 사용합니다.


### 1. 필요한 라이브러리 설치

`pip install -r requirements.txt`

### 2. 모델 학습

`script/train.sh`를 실행합니다


학습된 모델은 epoch 별로 `CHECKPOINT/epoch-{number}.bin` 으로 저장됩니다.<br>
Best Checkpoint가 `CHECKPOINT/best_model`에 저장됩니다.<br>

### 3. 추론하기

`script/test.sh`를 실행합니다


### 4. 제출하기

3번 스텝 `inference.py`에서 `RESULTDIR`에 저장된 `result.test.csv`와 `result.test2.csv`을 제출합니다.
