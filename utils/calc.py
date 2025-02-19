"""
Calculate various text features.
"""
import argparse
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import spacy
from textblob import TextBlob
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import textstat
from collections import Counter
import numpy as np
from scipy.stats import entropy

nlp = spacy.load('en_core_web_sm')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def calculate_token_entropy(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].squeeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    entropies = []
    for i in range(len(input_ids)):
        token_probs = probs[i, input_ids[i]]
        entropy_value = -torch.sum(token_probs * torch.log(token_probs + 1e-10)).item()
        entropies.append(entropy_value)
    return entropies

def calculate_perplexity(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].squeeze(0)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    log_likelihood = torch.sum(torch.log(probs + 1e-10), dim=-1)
    perplexity = torch.exp(-torch.mean(log_likelihood)).item()
    return perplexity

def calculate_lexical_diversity(text):
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0

def calculate_burstiness(text):
    sentences = text.split('.')
    sentence_lengths = [len(sentence.split()) for sentence in sentences]
    return np.var(sentence_lengths)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def calculate_readability(text):
    fk_score = textstat.flesch_kincaid_grade(text)  
    gf_score = textstat.gunning_fog(text)           
    return fk_score, gf_score

def calculate_pos_tag_distribution(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    return pos_counts

def calculate_ngram_distribution(text, n=2):
    tokens = word_tokenize(text)
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngram_counts = Counter(ngrams)
    return ngram_counts

def calculate_syntax_tree_depth(text):
    doc = nlp(text)
    depths = [len([token for token in sent.subtree]) for sent in doc.sents]
    return np.mean(depths)

def check_repetition(text, threshold=3):
    tokens = word_tokenize(text)
    token_counts = Counter(tokens)
    repeated_tokens = {token: count for token, count in token_counts.items() if count >= threshold}
    return repeated_tokens

def calc_main(args):
    dataset = load_dataset(args.dataset_name, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    for example in dataset:
        text = example[args.text_column]
        print(f"Processing text: {text[:100]}...")

        # Calculate features
        entropies = calculate_token_entropy(text, tokenizer, model)
        perplexity = calculate_perplexity(text, tokenizer, model)
        lexical_diversity = calculate_lexical_diversity(text)
        burstiness = calculate_burstiness(text)
        sentiment = analyze_sentiment(text)
        fk_score, gf_score = calculate_readability(text)
        pos_distribution = calculate_pos_tag_distribution(text)
        ngram_distribution = calculate_ngram_distribution(text)
        syntax_tree_depth = calculate_syntax_tree_depth(text)
        repetitions = check_repetition(text)

        # Print results
        print(f"Entropy: {np.mean(entropies)}")
        print(f"Perplexity: {perplexity}")
        print(f"Lexical Diversity (TTR): {lexical_diversity}")
        print(f"Burstiness (Variance in Sentence Length): {burstiness}")
        print(f"Sentiment: {sentiment}")
        print(f"Flesch-Kincaid Readability: {fk_score}")
        print(f"Gunning Fog Readability: {gf_score}")
        print(f"POS Tag Distribution: {pos_distribution}")
        print(f"N-gram Distribution: {ngram_distribution.most_common(5)}")
        print(f"Average Syntax Tree Depth: {syntax_tree_depth}")
        print(f"Repeated Tokens: {repetitions}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze text features to distinguish human vs AI-written text.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., "artem9k/ai-text-detection-pile").')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (default: "train").')
    parser.add_argument('--model_name', type=str, required=True, help='Pre-trained LLaMA model name (e.g., "meta-llama/Llama-2-7b-hf").')
    parser.add_argument('--text_column', type=str, default='text', help='Column containing the text data (default: "text").')

    args = parser.parse_args()
    calc_main(args)
