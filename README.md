# NLP-Project

Needs fixing:

`train.py`: unexpected stop of training after 1 epoch.

## Datasets

https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset

https://huggingface.co/datasets/artem9k/ai-text-detection-pile

## Installation

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Preprocessing

```
python utils/preprocess.py
```

![workflow](./assets/workflow.png)