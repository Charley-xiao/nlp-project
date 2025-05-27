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
import os

HF_DATASETS = ['artem9k/ai-text-detection-pile']
CSV_DATASETS = {
    'data/Training_Essay_Data.csv': ('text', 'generated'),
}

def preprocess(use_ghostbuster: bool = False) -> DataFrame:
    if not use_ghostbuster:
        all_data = []
        cnt_empty_text = 0
        for dataset_name in HF_DATASETS:
            dataset = load_dataset(dataset_name, split='train', cache_dir='cache')
            # dataset = dataset.select(range(1000))
            for entry in tqdm(dataset, desc=f"Processing {dataset_name}"):
                text = entry['text']
                if not text:
                    cnt_empty_text += 1
                    continue
                generated = 1 if entry['source'] == 'ai' else 0
                all_data.append({'text': text, 'generated': generated})
        print(f"Number of empty texts in HF_DATASETS: {cnt_empty_text}")
        
        for file_path, (text_column, generated_column) in CSV_DATASETS.items():
            csv_data = pd.read_csv(file_path)
            for _, row in tqdm(csv_data.iterrows(), desc=f"Processing {file_path}"):
                text = row[text_column]
                generated = row[generated_column]
                all_data.append({'text': text, 'generated': generated})
    else:
        claude_dir = 'data/ghostbuster-data/essay/claude'
        gpt_dir = 'data/ghostbuster-data/essay/gpt'
        human_dir = 'data/ghostbuster-data/essay/human'
        claude_files = [f for f in os.listdir(claude_dir) if f.endswith('.txt')]
        gpt_files = [f for f in os.listdir(gpt_dir) if f.endswith('.txt')]
        human_files = [f for f in os.listdir(human_dir) if f.endswith('.txt')]
        all_data = []
        cnt_empty_text = 0
        for file in tqdm(claude_files, desc="Processing Claude files"):
            with open(os.path.join(claude_dir, file), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if not text:
                    cnt_empty_text += 1
                    continue
                all_data.append({'text': text, 'generated': 1})
        for file in tqdm(gpt_files, desc="Processing GPT files"):
            with open(os.path.join(gpt_dir, file), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if not text:
                    cnt_empty_text += 1
                    continue
                all_data.append({'text': text, 'generated': 1})
        for file in tqdm(human_files, desc="Processing Human files"):
            with open(os.path.join(human_dir, file), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if not text:
                    cnt_empty_text += 1
                    continue
                all_data.append({'text': text, 'generated': 0})
        print(f"Number of empty texts in Ghostbuster data: {cnt_empty_text}")

    df = pd.DataFrame(all_data)
    df.to_csv('data/preprocessed.csv', index=False)

    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess text data for AI detection.")
    parser.add_argument('--use_ghostbuster', action='store_true')
    args = parser.parse_args()
    preprocess(args.use_ghostbuster)
