"""
Preprocess text data.

Final columns:
- text: Text data
- generated: 0 for human-written, 1 for AI-generated
"""
from datasets import load_dataset
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

HF_DATASETS = ['artem9k/ai-text-detection-pile']
CSV_DATASETS = {
    'data/Training_Essay_Data.csv': ('text', 'generated'),
}

def preprocess():
    all_data = []

    for dataset_name in HF_DATASETS:
        dataset = load_dataset(dataset_name, split='train', cache_dir='cache')
        # dataset = dataset.select(range(1000))
        for entry in tqdm(dataset, desc=f"Processing {dataset_name}"):
            text = entry['text']
            generated = 1 if entry['source'] == 'ai' else 0
            all_data.append({'text': text, 'generated': generated})
    
    for file_path, (text_column, generated_column) in CSV_DATASETS.items():
        csv_data = pd.read_csv(file_path)
        for _, row in tqdm(csv_data.iterrows(), desc=f"Processing {file_path}"):
            text = row[text_column]
            generated = row[generated_column]
            all_data.append({'text': text, 'generated': generated})

    df = pd.DataFrame(all_data)
    df.to_csv('data/preprocessed.csv', index=False)

    return df

if __name__ == '__main__':
    preprocess()
