## Baseline model of Goorm Project-3 NMT

��-> �� ������ ���� Encoder-Decoder ��

- �ѱ��� ������ ���� Decoder���� `monologg/koelectra-base-v3-discriminator`�� `token_embeddings`�� �ѱ��� pretrained Subword Embedding���� ����մϴ�.
- ���� �����͸� ���� Encoder���� `bert-base-uncased`�� pretrained model�� ����մϴ�.


### 1. �ʿ��� ���̺귯�� ��ġ

`pip install -r requirements.txt`

### 2. �� �н�

`script/train.sh`�� �����մϴ�


�н��� ���� epoch ���� `CHECKPOINT/epoch-{number}.bin` ���� ����˴ϴ�.<br>
Best Checkpoint�� `CHECKPOINT/best_model`�� ����˴ϴ�.<br>

### 3. �߷��ϱ�

`script/test.sh`�� �����մϴ�


### 4. �����ϱ�

3�� ���� `inference.py`���� `RESULTDIR`�� ����� `result.test.csv`�� `result.test2.csv`�� �����մϴ�.
