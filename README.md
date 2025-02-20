# NLP-Project

Needs fixing:

In `utils/gen_dataset.py`: 

1. `pos_distribution`, `ngram_distribution`, and `repetition` are not just numbers. Some preprocessing is needed.
2. `entropy_fft_features` is always a list of zeros. It should be fixed.
3. This error:
```
Traceback (most recent call last):
  File "D:\nlp-project\utils\gen_dataset.py", line 80, in <module>
    main(args)
  File "D:\nlp-project\utils\gen_dataset.py", line 66, in main
    for batch in dataloader:
                 ^^^^^^^^^^
  File "C:\Users\Charley\miniconda3\envs\py312\Lib\site-packages\torch\utils\data\dataloader.py", line 701, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\Charley\miniconda3\envs\py312\Lib\site-packages\torch\utils\data\dataloader.py", line 757, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Charley\miniconda3\envs\py312\Lib\site-packages\torch\utils\data\_utils\fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Charley\miniconda3\envs\py312\Lib\site-packages\torch\utils\data\_utils\collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Charley\miniconda3\envs\py312\Lib\site-packages\torch\utils\data\_utils\collate.py", line 172, in collate
    key: collate(
         ^^^^^^^^
  File "C:\Users\Charley\miniconda3\envs\py312\Lib\site-packages\torch\utils\data\_utils\collate.py", line 173, in collate
    [d[key] for d in batch], collate_fn_map=collate_fn_map
     ~^^^^^
KeyError: 'you'
```

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