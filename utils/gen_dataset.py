import torch
from torch.utils.data import Dataset, DataLoader
from utils.calc import (
    text_to_handcrafted_features
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd 
import numpy as np
import argparse
import os
import random
import pickle
from tqdm import tqdm
import concurrent.futures

class TextClassificationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        handcrafted_features, text, label = self.dataset[idx]
        return np.float32(handcrafted_features), text, label

def generate_dataset(csv_file, model_name, num_fft_features=10):
    """
    Generate train, validation, and test datasets from a CSV file.

    Args:
    - csv_file (str): Path to the CSV file containing the cleaned dataset, containing only two columns: 'text' and 'generated'.
    - model_name (str): Pretrained model name.
    - num_fft_features (int): The number of FFT features to extract from entropy values.

    Returns:
    - train_dataset (TextClassificationDataset): Training dataset.
    - val_dataset (TextClassificationDataset): Validation dataset.
    - test_dataset (TextClassificationDataset): Test dataset.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    if (os.path.exists('cache')
        and os.path.exists('cache/train.pkl') 
        and os.path.exists('cache/val.pkl') 
        and os.path.exists('cache/test.pkl')):
        print("Loading cached dataset...")
        with open('cache/train.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        with open('cache/val.pkl', 'rb') as f:
            val_dataset = pickle.load(f)
        with open('cache/test.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        dataset = []
        cache_contains_temp_pkl = False
        if os.path.exists('cache'):
            for file in os.listdir('cache'):
                if file.startswith('temp_') and file.endswith('.pkl'):
                    cache_contains_temp_pkl = True
                    break
        if cache_contains_temp_pkl:
            print("[WARNING] Loading cached partial dataset...")
            print("[WARNING] If you want to regenerate the dataset, delete the cache folder.")
            for file in os.listdir('cache'):
                if file.startswith('temp_') and file.endswith('.pkl'):
                    with open(os.path.join('cache', file), 'rb') as f:
                        dataset.extend(pickle.load(f))
        else:
            print("Generating dataset...")
            df = pd.read_csv(csv_file)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Device: {device}")
            model.to(device)
            # For each text, calculate handcrafted features
            print("Calculating handcrafted features... Total rows:", len(df))
            for _, row in tqdm(df.iterrows()):
                text = row['text']
                try:
                    handcrafted_features = text_to_handcrafted_features(text, tokenizer, model, num_fft_features)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Text: {text}")
                    continue
                label = row['generated']
                dataset.append((handcrafted_features, text, label))
                # Save dataset every 1000 rows
                if len(dataset) % 1000 == 0:
                    try:
                        # Remove previous cache files
                        for file in os.listdir('cache'):
                            if file.startswith('temp_') and file.endswith('.pkl'):
                                os.remove(os.path.join('cache', file))
                    except Exception as e:
                        print(f"Error removing cache files: {e}")
                    with open(f'cache/temp_{len(dataset)}.pkl', 'wb') as f:
                        pickle.dump(dataset, f)
        random.shuffle(dataset)
        # Split dataset into train, validation, and test sets
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size+val_size]
        test_dataset = dataset[train_size+val_size:]
        # Cache datasets
        if not os.path.exists('cache'):
            os.makedirs('cache')
        with open('cache/train.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open('cache/val.pkl', 'wb') as f:
            pickle.dump(val_dataset, f)
        with open('cache/test.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
    train_dataset = TextClassificationDataset(train_dataset)
    val_dataset = TextClassificationDataset(val_dataset)
    test_dataset = TextClassificationDataset(test_dataset)
    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description="Generate dataset for text classification")
    parser.add_argument("--csv_file", type=str, default="data/preprocessed.csv", help="Path to preprocessed CSV file")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Pretrained model name")
    parser.add_argument("--num_fft_features", type=int, default=10, help="Number of FFT features")
    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset = generate_dataset(args.csv_file, args.model_name, args.num_fft_features)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print("Finished generating dataset.")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    for handcrafted_features, text, label in train_loader:
        print(handcrafted_features.size(), len(text), label)
        break
    for handcrafted_features, text, label in val_loader:
        print(handcrafted_features.size(), len(text), label)
        break
    for handcrafted_features, text, label in test_loader:
        print(handcrafted_features.size(), len(text), label)
        break