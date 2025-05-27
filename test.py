import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

from model import ClassifierBackbone 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from utils.gen_dataset import TextClassificationDataset
from utils.calc import (
    text_to_handcrafted_features
)


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    encoder_model = AutoModel.from_pretrained(args.encoder_model_name).to(device)
    encoder_model.eval()
    latent_dim = encoder_model.config.hidden_size

    entropy_tokenizer = AutoTokenizer.from_pretrained(args.entropy_model_name)
    entropy_model = AutoModelForCausalLM.from_pretrained(args.entropy_model_name).to(device)
    entropy_model.eval()
    entropy_tokenizer.pad_token = entropy_tokenizer.eos_token

    test_dataset = []
    if "ghostbuster" in args.test_data_path:
        claude_dir = os.path.join(args.test_data_path, "essay", "claude")
        claude_files = [os.path.join(claude_dir, f) for f in os.listdir(claude_dir) if f.endswith('.txt')]
        print(f"Found {len(claude_files)} Claude files for testing.")
        gpt_dir = os.path.join(args.test_data_path, "essay", "gpt")
        gpt_files = [os.path.join(gpt_dir, f) for f in os.listdir(gpt_dir) if f.endswith('.txt')]
        print(f"Found {len(gpt_files)} GPT files for testing.")
        human_dir = os.path.join(args.test_data_path, "essay", "human")
        human_files = [os.path.join(human_dir, f) for f in os.listdir(human_dir) if f.endswith('.txt')]
        print(f"Found {len(human_files)} human files for testing.")
        for file in tqdm(claude_files, desc="Processing Claude files"):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                handcrafted_features = text_to_handcrafted_features(text, entropy_model, entropy_tokenizer, device)
                test_dataset.append((handcrafted_features, text, 1))
        for file in tqdm(gpt_files, desc="Processing GPT files"):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                handcrafted_features = text_to_handcrafted_features(text, entropy_model, entropy_tokenizer, device)
                test_dataset.append((handcrafted_features, text, 1))
        for file in tqdm(human_files, desc="Processing Human files"):
            with open(file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                handcrafted_features = text_to_handcrafted_features(text, entropy_model, entropy_tokenizer, device)
                test_dataset.append((handcrafted_features, text, 0))
        print(f"Total test dataset size: {len(test_dataset)}")
        test_dataset = TextClassificationDataset(test_dataset)

    model = ClassifierBackbone(
        args.handcrafted_dim,
        latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward
    ).to(device)
    print("Model initialized:")
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    if os.path.exists(args.test_checkpoint):
        model.load_state_dict(torch.load(args.test_checkpoint, weights_only=True, map_location=device))
        print(f"Loaded model from {args.test_checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint file {args.test_checkpoint} not found.")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for handcrafted_features, texts, labels in test_loader:
            handcrafted_features = handcrafted_features.to(device)
            labels = labels.to(device)
            latent_features = encoder_model(**encoder_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)).last_hidden_state.mean(dim=1)
            latent_features = latent_features.to(device)
            logits = model(handcrafted_features, latent_features)

            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Train binary classifier for human vs machine text")
    parser.add_argument("--handcrafted_dim", type=int, default=21, help="Dimension of handcrafted features")
    parser.add_argument("--entropy_model_name", type=str, default="gpt2", help="Pretrained entropy model name")
    parser.add_argument("--encoder_model_name", type=str, default="roberta-base", help="Pretrained encoder model name")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for classifier backbone")
    parser.add_argument("--output_dim", type=int, default=2, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=10, help="Number of layers in the transformer")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of the feedforward network")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--test_checkpoint", type=str, default="checkpoints/classifier.pt", help="Path to classifier checkpoint for testing")
    parser.add_argument("--test_data_path", type=str, default="data/test_data.json", help="Path to test data JSON file")
    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
