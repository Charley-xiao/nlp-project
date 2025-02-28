from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from model import ClassifierBackbone
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import os
import argparse

from utils.gen_dataset import text_to_handcrafted_features

argparser = argparse.ArgumentParser()
argparser.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pt", help="Path to classifier checkpoint")
argparser.add_argument("--report_path", type=str, default="assets/report.md", help="Path to report markdown file")
argparser.add_argument("--handcrafted_dim", type=int, default=21, help="Dimension of handcrafted features")
argparser.add_argument("--entropy_model_name", type=str, default="gpt2", help="Pretrained entropy model name")
argparser.add_argument("--encoder_model_name", type=str, default="roberta-base", help="Pretrained encoder model name")
argparser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for classifier backbone")
argparser.add_argument("--output_dim", type=int, default=2, help="Number of output classes")
argparser.add_argument("--dropout", type=float, default=0, help="Dropout rate")
argparser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
argparser.add_argument("--num_layers", type=int, default=10, help="Number of layers in the transformer")
argparser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of the feedforward network")
argparser.add_argument("--model_version", type=str, default="v0.1", help="Model version")
argparser.add_argument("--grpo_dataset", type=str, default="intone/horror_stories_reddit", help="Path to GRPO dataset")
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
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.eval()
    entropy_model = AutoModelForCausalLM.from_pretrained(args.entropy_model_name).to(device)
    entropy_model.eval()
    entropy_tokenizer = AutoTokenizer.from_pretrained(args.entropy_model_name)

    return encoder_tokenizer, encoder_model, classifier, entropy_tokenizer, entropy_model

encoder_tokenizer, encoder_model, classifier, entropy_tokenizer, entropy_model = load_models(args)

@torch.no_grad()
def predict(text: str) -> int:
    """
    Runs the text through the classifier backbone (handcrafted + latent features).
    Returns 0 for "human-like" or 1 for "AI-like" class.
    """
    handcrafted_features = text_to_handcrafted_features(text)
    handcrafted_features = torch.tensor(handcrafted_features).unsqueeze(0).to(device)
    tokenized_inputs = encoder_tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    latent_features = encoder_model(**tokenized_inputs).last_hidden_state.mean(dim=1).to(device)
    logits = classifier(handcrafted_features, latent_features)
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

def calc_reward(text: str) -> float:
    """
    Converts classifier prediction (0 or 1) into reward.
    Here, if the classifier says 'human-like' = 0 => reward=1,
    if 'AI-like' = 1 => reward=0.
    """
    prediction = predict(text)
    return 1 - prediction  # (If AI-like => 0, if human-like => 1)

def split_text_into_prompt_and_completion(example):
    """
    Takes a dataset example with 'title' and 'text'.
    Splits the text into the first paragraph (for the prompt)
    and the rest (for the completion).
    """
    title = example["title"]
    full_text = example["text"]

    paragraphs = full_text.split("\n\n")
    if len(paragraphs) <= 1:
        first_paragraph = full_text.strip()
        remainder = ""
    else:
        first_paragraph = paragraphs[0].strip()
        remainder = "\n\n".join(paragraphs[1:]).strip()

    prompt = (
        "You are a creative and skilled horror storyteller. "
        "Continue the story using a human-like style.\n\n"
        f"Title: {title}\n"
        "First Paragraph:\n"
        f"{first_paragraph}\n\n"
        "### Instruction: Continue the story.\n"
        "### Response:"
    )

    example["prompt"] = prompt
    example["completion"] = remainder

    return example

if args.grpo_dataset.endswith(".csv"):
    dataset = load_dataset("csv", data_files=args.grpo_dataset)["train"]
else:
    dataset = load_dataset(args.grpo_dataset, split="train")

dataset = dataset.map(split_text_into_prompt_and_completion)

training_args = GRPOConfig(
    output_dir="Qwen2.5-0.5B-GRPO",
    logging_steps=10
)
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=calc_reward,
    args=training_args,
    train_dataset=dataset,
    prompt_column="prompt",
    response_column="completion",
)
trainer.train()
