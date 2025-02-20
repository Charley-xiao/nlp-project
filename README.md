<div align="center">
  <h1>AI Text Detection</h1>
  <p>Project for CS310 NLP, Spring 2025 at SUSTech</p>

[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](TODO)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Charley-xiao/nlp-project)](https://github.com/Charley-xiao/nlp-project/commits/master)
[![GitHub issues](https://img.shields.io/github/issues/Charley-xiao/nlp-project)](https://github.com/Charley-xiao/nlp-project/issues)
</div>

> Needs fixing: `train.py`: unexpected stop of training after 1 epoch.

## Overview

<div align="center">
  <img src="./assets/workflow.png" alt="overview" width="80%">
</div>

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