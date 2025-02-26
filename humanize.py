from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from model import ClassifierBackbone
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from utils.gen_dataset import text_to_handcrafted_features
import os 
import argparse
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pt", help="Path to classifier checkpoint")
argparser.add_argument("--report_path", type=str, default="assets/report.md", help="Path to report markdown file")
argparser.add_argument("--handcrafted_dim", type=int, default=21, help="Dimension of handcrafted features")
argparser.add_argument("--entropy_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Pretrained entropy model name")
argparser.add_argument("--encoder_model_name", type=str, default="roberta-base", help="Pretrained encoder model name")
argparser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for classifier backbone")
argparser.add_argument("--output_dim", type=int, default=2, help="Number of output classes")
argparser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
argparser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
argparser.add_argument("--num_layers", type=int, default=10, help="Number of layers in the transformer")
argparser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of the feedforward network")
argparser.add_argument("--model_version", type=str, default="v0.1", help="Model version")
argparser.add_argument("--grpo_dataset", type=str, default="grpo_dataset", help="Path to GRPO dataset")
args = argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(args):
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    encoder_model = AutoModel.from_pretrained(args.encoder_model_name).to(device)
    encoder_model.eval()
    latent_dim = encoder_model.config.hidden_size

    classifier = ClassifierBackbone(
        args.handcrafted_dim,
        latent_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        dropout=args.dropout,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward
    ).to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, weights_only=True))
    classifier.eval()

    entropy_model = AutoModelForCausalLM.from_pretrained(args.entropy_model_name).to(device)
    entropy_model.eval()
    entropy_tokenizer = AutoTokenizer.from_pretrained(args.entropy_model_name)

    return encoder_tokenizer, encoder_model, classifier, entropy_tokenizer, entropy_model

encoder_tokenizer, encoder_model, classifier, entropy_tokenizer, entropy_model = load_models(args)

@torch.no_grad()
def predict(text):
    handcrafted_features = text_to_handcrafted_features(text)
    handcrafted_features = torch.tensor(handcrafted_features).unsqueeze(0).to(device)
    texts = [text]

    latent_features = encoder_model(**encoder_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)).last_hidden_state.mean(dim=1)
    latent_features = latent_features.to(device)
    logits = classifier(handcrafted_features, latent_features)
    prediction = torch.argmax(logits, dim=1).item()

    return prediction

def calc_reward(text):
    prediction = predict(text)
    return 1 - prediction # FUTURE: consider length of text

if args.grpo_dataset.endswith(".csv"):
    dataset = load_dataset("csv", data_files=args.grpo_dataset)
else:
    dataset = load_dataset(args.grpo_dataset, split="train")
training_args = GRPOConfig(output_dir="Qwen2.5-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=calc_reward,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()