# NL2SQL Finetune
GPT 모델을 활용한 자연어 -> SQL 생성 작업

## Install
- Ubuntu 18.04
- Python 3.11.4
```bash
pip install -r requirements.txt
```

## Dataset
- [AI-Hub 자연어 기반 질의(NL2SQL) 검색 생성 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=ty&dataSetSn=71351)
- 24.2MB
- 107,860개

![NL2SQL Data Example](AI-Hub_NL2SQL_Data_example.png)

### Preprocess Dataset
[nl2sql.json](./dataset/nl2sql/nl2sql.json)

## Train
```bash
python train.py
```

### Pretrained Model
- [beomi/llama-2-ko-7b](https://huggingface.co/beomi/llama-2-ko-7b)
    - 성능 안나옴
- [hyunseoki/ko-en-llama2-13b](https://huggingface.co/hyunseoki/ko-en-llama2-13b)
    - 성능 안나옴
- [beomi/llama-2-koen-13b](https://huggingface.co/beomi/llama-2-koen-13b)
    - 학습 예정
- [beomi/llama-2-ko-70b](https://huggingface.co/beomi/llama-2-ko-70b)
    - A6000 (48GB), 1 batch => OOM


## Inference
```bash
python inference.py
```

## Performance
![Performance Example](Performance.png)

- Column, Table명 튜닝이 안됨
- `AND`, `OR`, `LIKE`, 부등호 문법 오류 발생

## Reference
- [Llama 2 Fine-Tune with QLoRA](https://youtu.be/eeM6V5aPjhk?si=f_9LM0JmDTe2jlx1)
- [RYANSQL](https://github.com/kakaoenterprise/RYANSQL)
