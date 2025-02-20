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
    pass