"""
Unit tests for text features, excluding the functions that require a trained model.
"""
import unittest
from collections import Counter
import numpy as np
from textblob import TextBlob
import textstat
import spacy
import nltk
from nltk.tokenize import word_tokenize
from utils import (
    calculate_lexical_diversity,
    calculate_burstiness,
    analyze_sentiment,
    calculate_readability,
    calculate_pos_tag_distribution,
    calculate_ngram_distribution,
    calculate_syntax_tree_depth,
    check_repetition,
)

nlp = spacy.load('en_core_web_sm')

class TestTextFeatures(unittest.TestCase):
    
    def setUp(self):
        self.text = "The quick brown fox jumps over the lazy dog. It is a well-known sentence."
        self.short_text = "Hello world!"
        
    def test_calculate_lexical_diversity(self):
        result = calculate_lexical_diversity(self.text)
        tokens = word_tokenize(self.text)
        unique_tokens = set(tokens)
        expected_result = len(unique_tokens) / len(tokens) if len(tokens) > 0 else 0
        self.assertAlmostEqual(result, expected_result, places=2)
        
    def test_calculate_burstiness(self):
        result = calculate_burstiness(self.text)
        sentences = self.text.split('.')
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        expected_result = np.var(sentence_lengths)
        self.assertAlmostEqual(result, expected_result, places=2)
        
    def test_analyze_sentiment(self):
        result = analyze_sentiment(self.text)
        blob = TextBlob(self.text)
        expected_result = blob.sentiment.polarity
        self.assertAlmostEqual(result, expected_result, places=2)
        
    def test_calculate_readability(self):
        result = calculate_readability(self.text)
        fk_score = textstat.flesch_kincaid_grade(self.text)
        gf_score = textstat.gunning_fog(self.text)
        self.assertAlmostEqual(result[0], fk_score, places=2)
        self.assertAlmostEqual(result[1], gf_score, places=2)
        
    def test_calculate_pos_tag_distribution(self):
        result = calculate_pos_tag_distribution(self.text)
        tokens = word_tokenize(self.text)
        tagged_tokens = nltk.pos_tag(tokens)
        expected_result = Counter(tag for word, tag in tagged_tokens)
        self.assertEqual(result, expected_result)
        
    def test_calculate_ngram_distribution(self):
        result = calculate_ngram_distribution(self.text, n=2)
        tokens = word_tokenize(self.text)
        bigrams = zip(*[tokens[i:] for i in range(2)])
        expected_result = Counter(bigrams)
        self.assertEqual(result, expected_result)
        
    def test_calculate_syntax_tree_depth(self):
        result = calculate_syntax_tree_depth(self.text)
        doc = nlp(self.text)
        depths = [len([token for token in sent.subtree]) for sent in doc.sents]
        expected_result = np.mean(depths)
        self.assertAlmostEqual(result, expected_result, places=2)
        
    def test_check_repetition(self):
        result = check_repetition(self.text, threshold=2)
        token_counts = Counter(word_tokenize(self.text))
        expected_result = {token: count for token, count in token_counts.items() if count >= 2}
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()
