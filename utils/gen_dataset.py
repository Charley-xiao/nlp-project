import torch
from torch.utils.data import Dataset, DataLoader
from calc import (
    calculate_lexical_diversity,
    calculate_burstiness,
    analyze_sentiment,
    calculate_readability,
    calculate_pos_tag_distribution,
    calculate_ngram_distribution,
    calculate_syntax_tree_depth,
    check_repetition,
    calculate_token_entropy,
    compute_entropy_fft_features
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd 
import numpy as np
import argparse

class TextAndFeaturesDataset(Dataset):
    def __init__(self, data, text_column, model_name, fft_features=10):
        self.data = data
        self.text_column = text_column
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.fft_features = fft_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx][self.text_column]
        generated = self.data.iloc[idx]['generated']
        entropies = calculate_token_entropy(text, self.tokenizer, self.model)
        entropy_fft_features = compute_entropy_fft_features(entropies, num_features=self.fft_features)
        lexical_diversity = calculate_lexical_diversity(text)
        burstiness = calculate_burstiness(text)
        sentiment = analyze_sentiment(text)
        fk_score, gf_score = calculate_readability(text)
        pos_distribution = calculate_pos_tag_distribution(text)
        ngram_distribution = calculate_ngram_distribution(text)
        syntax_tree_depth = calculate_syntax_tree_depth(text)
        repetition = check_repetition(text)
        return {
            'text': text,
            'generated': generated,
            'lexical_diversity': lexical_diversity,
            'burstiness': burstiness,
            'sentiment': sentiment,
            'fk_score': fk_score,
            'gf_score': gf_score,
            'pos_distribution': pos_distribution,
            'ngram_distribution': ngram_distribution,
            'syntax_tree_depth': syntax_tree_depth,
            'repetition': repetition,
            'entropy_fft_features': entropy_fft_features
        }
    
def main(args):
    data = pd.read_csv(args.dataset_path)
    dataset = TextAndFeaturesDataset(data, args.text_column, args.model_name, args.fft_features)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for batch in dataloader:
        print(batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset with text features")
    parser.add_argument("--dataset_path", default="data/preprocessed.csv", type=str, help="Path to the input dataset CSV file")
    parser.add_argument("--text_column", default="text", type=str, help="Name of the column containing text data")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct", type=str, help="Name of the pre-trained model")
    parser.add_argument("--fft_features", type=int, default=10, help="Number of FFT features to compute")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    args = parser.parse_args()
    main(args)