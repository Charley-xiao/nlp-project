import os
import sys
import nltk
import spacy
import streamlit as st
import random
import argparse
import numpy as np
from model import ClassifierBackbone
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
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

st.set_page_config(page_title="VeriScribbi: Text Source Identifier", layout="centered")
st.title("VeriScribbi: Text Source Identifier")
classifier_tab, report_tab = st.tabs(["Classifier", "Report"])

@st.cache_resource
def load_models(_args):
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        spacy.load("en_core_web_sm")

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    encoder_model = AutoModel.from_pretrained(args.encoder_model_name)
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
    )
    classifier.load_state_dict(torch.load(args.classifier_path, weights_only=True))
    classifier.eval()

    entropy_model = AutoModelForCausalLM.from_pretrained(args.entropy_model_name)
    entropy_tokenizer = AutoTokenizer.from_pretrained(args.entropy_model_name)

    return encoder_tokenizer, encoder_model, classifier, entropy_tokenizer, entropy_model

encoder_tokenizer, encoder_model, classifier, entropy_tokenizer, entropy_model = load_models(args)

with classifier_tab:
    st.header("Distinguish Human vs. Machine Written Text")
    input_text = st.text_area("Enter your text here:")

    if st.button("Classify"):
        if input_text.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner('Classifying text...'):
                try:
                    handcrafted_features = text_to_handcrafted_features(input_text, entropy_tokenizer, entropy_model)
                    handcrafted_features = torch.tensor(np.float32(handcrafted_features)).unsqueeze(0)
                    latent_features = encoder_model(
                        **encoder_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                    ).last_hidden_state.mean(dim=1)
                except IndexError:
                    st.error("The text is too short to classify. Please try a longer text.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                try:
                    logits = classifier(handcrafted_features, latent_features)
                    prediction = torch.argmax(logits, dim=1).item()
                    prob = torch.softmax(logits, dim=1).max().item() * 100

                    st.subheader("Classification Result")

                    st.write("""
                    <style>
                    .result-card {
                        background-color: #F9F9F9;
                        border-radius: 10px;
                        padding: 20px;
                        margin-top: 15px;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    }
                    .result-label {
                        font-size: 1.3rem;
                        font-weight: bold;
                        margin-bottom: 5px;
                    }
                    .result-prob {
                        font-size: 1rem;
                        color: #555;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    if prediction == 1:
                        label_text = "🤖 Machine Generated"
                        label_color = "#FF4B4B"
                    else:
                        label_text = "🙋 Human Written"
                        label_color = "#2ECC71"

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label" style="color: {label_color};">
                            {label_text}
                        </div>
                        <div class="result-prob">
                            Probability: {prob:.2f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

with report_tab:
    st.header("Report")
    with open(args.report_path, "r") as file:
        report = file.read()
    st.markdown(report)
