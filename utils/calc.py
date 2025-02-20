"""
Calculate various text features.
"""
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
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
    entropies_tensor = - torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    entropies = entropies_tensor.numpy().tolist()
    return entropies

def compute_entropy_fft_features(entropy_values, num_features=10):
    """
    Compute FFT on a sequence of per-token entropy values and extract a fixed number of features.
    Args:
        entropy_values (list or np.array): The per-token entropy values.
        num_features (int): The number of FFT coefficients (magnitudes) to extract.
    Returns:
        np.array: A feature vector of length `num_features` representing the frequency domain.
    """
    entropy_array = np.array(entropy_values)
    fft_vals = np.fft.fft(entropy_array)
    fft_magnitudes = np.abs(fft_vals)[:len(fft_vals)//2]
    if len(fft_magnitudes) < num_features:
        fft_features = np.pad(fft_magnitudes, (0, num_features - len(fft_magnitudes)), mode='constant')
    else:
        fft_features = fft_magnitudes[:num_features]
    return fft_features

def calculate_perplexity(texts, tokenizer, model):
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs['input_ids']  # [batch_size, seq_len]
    attention_mask = inputs.get('attention_mask', None)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits: [batch_size, seq_len, vocab_size]
        logits = outputs.logits
    # log_softmax => [batch_size, seq_len, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)
    real_token_log_probs = torch.gather(log_probs, 2, input_ids.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        valid_log_probs = real_token_log_probs * attention_mask
        n_valid_tokens = attention_mask.sum().item()
    else:
        valid_log_probs = real_token_log_probs
        n_valid_tokens = real_token_log_probs.numel()
    nll = - valid_log_probs
    avg_nll = nll.sum() / n_valid_tokens
    ppl = torch.exp(avg_nll)
    return ppl.item()

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

    for example in dataset.select(range(5)):
        text = example[args.text_column]
        print(f"Processing text: {text[:100]}...")

        # Calculate per-token entropy and extract FFT features from it
        entropies = calculate_token_entropy(text, tokenizer, model)
        entropy_fft_features = compute_entropy_fft_features(entropies, num_features=args.fft_features)
        
        # Calculate additional features
        perplexity = calculate_perplexity(text, tokenizer, model)
        lexical_diversity = calculate_lexical_diversity(text)
        burstiness = calculate_burstiness(text)
        sentiment = analyze_sentiment(text)
        fk_score, gf_score = calculate_readability(text)
        pos_distribution = calculate_pos_tag_distribution(text)
        ngram_distribution = calculate_ngram_distribution(text)
        syntax_tree_depth = calculate_syntax_tree_depth(text)
        repetitions = check_repetition(text)

        # Print (or save) results. For a classifier, you might want to aggregate these into a single feature vector.
        print(f"Entropy Values: {entropies}")
        print(f"Average Entropy: {np.mean(entropies):.4f}")
        print(f"Entropy FFT Features: {entropy_fft_features}")
        print(f"Perplexity: {perplexity:.4f}")
        print(f"Lexical Diversity (TTR): {lexical_diversity:.4f}")
        print(f"Burstiness (Variance in Sentence Length): {burstiness:.4f}")
        print(f"Sentiment: {sentiment:.4f}")
        print(f"Flesch-Kincaid Readability: {fk_score:.4f}")
        print(f"Gunning Fog Readability: {gf_score:.4f}")
        print(f"POS Tag Distribution: {pos_distribution}")
        print(f"N-gram Distribution (Top 5): {ngram_distribution.most_common(5)}")
        print(f"Average Syntax Tree Depth: {syntax_tree_depth:.4f}")
        print(f"Repeated Tokens: {repetitions}")
        print("-" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze text features to distinguish human vs AI-written text.")
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (e.g., "artem9k/ai-text-detection-pile").')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (default: "train").')
    parser.add_argument('--model_name', type=str, required=True, help='Pre-trained LLaMA model name (e.g., "meta-llama/Llama-2-7b-hf").')
    parser.add_argument('--text_column', type=str, default='text', help='Column containing the text data (default: "text").')
    parser.add_argument('--fft_features', type=int, default=10, help='Number of FFT features to extract from entropy values (default: 10).')

    args = parser.parse_args()
    calc_main(args)
