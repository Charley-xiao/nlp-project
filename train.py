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

from model import ClassifierBackbone 
from utils.gen_dataset import TextClassificationDataset, generate_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


def train_and_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = AutoTokenizer.from_pretrained(args.entropy_model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # model = AutoModelForCausalLM.from_pretrained(args.entropy_model_name)
    # model.to(device)

    train_dataset, val_dataset, test_dataset = generate_dataset(args.dataset_csv, args.entropy_model_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    encoder_model = AutoModel.from_pretrained(args.encoder_model_name)
    latent_dim = encoder_model.config.hidden_size

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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epoch in trange(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for i, (handcrafted_features, texts, labels) in enumerate(train_loader):
            handcrafted_features = handcrafted_features.to(device)
            labels = labels.to(device)

            latent_features = encoder_model(**encoder_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)).last_hidden_state.mean(dim=1)
            latent_features = latent_features.to(device)

            optimizer.zero_grad()
            logits = model(handcrafted_features, latent_features)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            optimizer.step()

            running_loss += loss.item() * handcrafted_features.size(0)

            if (i + 1) % args.log_interval == 0:
                print(f"Epoch [{epoch}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch}/{args.epochs}], Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for handcrafted_features, texts, labels in val_loader:
                handcrafted_features = handcrafted_features.to(device)
                labels = labels.to(device)
                latent_features = encoder_model(**encoder_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)).last_hidden_state.mean(dim=1)
                latent_features = latent_features.to(device)
                logits = model(handcrafted_features, latent_features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * handcrafted_features.size(0)

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_dataset)
        accuracy = correct / total
        print(f"Epoch [{epoch}/{args.epochs}], Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        if epoch % args.save_interval == 0:
            checkpoint_path = f"checkpoints/{args.checkpoint_prefix}_epoch{epoch}.pt"
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")


    torch.save(model.state_dict(), f"checkpoints/classifier.pt")
    print("Training complete. Model saved.")

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for handcrafted_features, texts, labels in test_loader:
            handcrafted_features = handcrafted_features.to(device)
            labels = labels.to(device)
            latent_features = encoder_model(**encoder_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)).last_hidden_state.mean(dim=1)
            latent_features = latent_features.to(device)
            logits = model(handcrafted_features, latent_features)
            loss = criterion(logits, labels)
            test_loss += loss.item() * handcrafted_features.size(0)

            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= len(test_dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Train binary classifier for human vs machine text")
    parser.add_argument("--dataset_csv", type=str, default="data/preprocessed.csv", help="Path to training CSV file")
    parser.add_argument("--handcrafted_dim", type=int, default=21, help="Dimension of handcrafted features")
    parser.add_argument("--entropy_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Pretrained entropy model name")
    parser.add_argument("--encoder_model_name", type=str, default="roberta-base", help="Pretrained encoder model name")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for classifier backbone")
    parser.add_argument("--output_dim", type=int, default=2, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--nhead", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the transformer")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of the feedforward network")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--step_size", type=int, default=5, help="Step size for the LR scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Gamma for the LR scheduler")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--log_interval", type=int, default=10, help="Log training status every N steps")
    parser.add_argument("--save_interval", type=int, default=5, help="Save a checkpoint every N epochs")
    parser.add_argument("--checkpoint_prefix", type=str, default="classifier_checkpoint", help="Prefix for checkpoint filenames")
    args = parser.parse_args()
    train_and_test(args)


if __name__ == "__main__":
    main()
