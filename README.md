<div align="center">
  <h1>AI Text Detection</h1>
  <p>Project for CS310 NLP, Spring 2025 at SUSTech</p>

[![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](TODO)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Charley-xiao/nlp-project)](https://github.com/Charley-xiao/nlp-project/commits/master)
[![GitHub issues](https://img.shields.io/github/issues/Charley-xiao/nlp-project)](https://github.com/Charley-xiao/nlp-project/issues)
</div>

> Needs fixing: `train.py`: validation acc.

## Overview

<div align="center">
  <img src="./assets/workflow.png" alt="overview" width="80%">
</div>

TODO: Add description

We released a [web app](TODO) and also made the checkpoints for the model available at [where?](TODO).

## Datasets

We combined two datasets for training and testing:
1. [LLM - Detect AI Generated Text Dataset](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset)
2. [AI Text Dectection Pile](https://huggingface.co/datasets/artem9k/ai-text-detection-pile)

TODO: Add dataset description

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python utils/download.py
python utils/preprocess.py
```

The last command will preprocess the datasets and save the combined dataset in the `data/` directory.

## Training

```bash
python train.py
```

TODO: Add training description

## Evaluation

```bash
TODO
```

TODO: Add evaluation description

## Results

See TODO.

## Contributors

![[](https://github.com/Charley-xiao/nlp-project/graphs/contributors)](https://contrib.rocks/image?repo=Charley-xiao/nlp-project)

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.