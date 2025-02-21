import streamlit as st
import os
import sys
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
os.system(f'sudo {sys.executable} -m spacy download en_core_web_sm')
import random
import argparse
from model import ClassifierBackbone
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.gen_dataset import text_to_handcrafted_features

torch.classes.__path__ = []

argparser = argparse.ArgumentParser()
argparser.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pt", help="Path to classifier checkpoint")
argparser.add_argument("--report_path", type=str, default="assets/report.md", help="Path to report markdown file")
argparser.add_argument("--handcrafted_dim", type=int, default=21, help="Dimension of handcrafted features")
argparser.add_argument("--entropy_model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Pretrained entropy model name")
argparser.add_argument("--encoder_model_name", type=str, default="roberta-base", help="Pretrained encoder model name")
argparser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for classifier backbone")
argparser.add_argument("--output_dim", type=int, default=2, help="Number of output classes")
argparser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
argparser.add_argument("--nhead", type=int, default=2, help="Number of attention heads")
argparser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the transformer")
argparser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of the feedforward network")
argparser.add_argument("--model_version", type=str, default="v0.1", help="Model version")
args = argparser.parse_args()

if not os.path.exists(args.classifier_path):
    os.makedirs("checkpoints", exist_ok=True)
    commands = ['sudo', 'wget', f'https://github.com/Charley-xiao/nlp-project/releases/download/{args.model_version}/classifier.tar.gz', '-O', 'checkpoints/classifier.tar.gz']
    os.system(' '.join(commands))
    commands = ['sudo', 'tar', '-xvf', 'checkpoints/classifier.tar.gz', '-C', 'checkpoints']
    os.system(' '.join(commands))


classifier = ClassifierBackbone(
    args.handcrafted_dim,
    args.encoder_model_name,
    hidden_dim=args.hidden_dim,
    output_dim=args.output_dim,
    dropout=args.dropout,
    nhead=args.nhead,
    num_layers=args.num_layers,
    dim_feedforward=args.dim_feedforward
)
classifier.load_state_dict(torch.load(args.classifier_path, map_location=torch.device('cpu')))
classifier.eval()

entropy_model = AutoModelForCausalLM.from_pretrained(args.entropy_model_name)
entropy_tokenizer = AutoTokenizer.from_pretrained(args.entropy_model_name)

st.set_page_config(page_title="Text Source Identifier", layout="centered")
st.title("Text Source Identifier")
classifier_tab, report_tab = st.tabs(["Classifier", "Report"])

with classifier_tab:
    st.header("Distinguish Human vs. Machine Written Text")
    input_text = st.text_area("Enter your text here:")

    if st.button("Classify"):
        if input_text.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            handcrafted_features = text_to_handcrafted_features(input_text, entropy_tokenizer, entropy_model)
            logits = classifier(torch.tensor(handcrafted_features).unsqueeze(0), [input_text])
            prediction = torch.argmax(logits, dim=1).item()
            result = "Machine Generated Text" if prediction == 1 else "Human Written Text"
            st.success(f"Prediction: {result}")

with report_tab:
    st.header("Report")
    with open(args.report_path, "r") as file:
        report = file.read()
    st.markdown(report)
