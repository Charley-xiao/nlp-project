import os
import pandas as pd
import unittest

class TestPreprocessedCSV(unittest.TestCase):
    
    def test_preprocessed_csv_format(self):
        preprocessed_file = 'data/preprocessed.csv'
        self.assertTrue(os.path.exists(preprocessed_file), "Preprocessed CSV file does not exist.")
        
        df = pd.read_csv(preprocessed_file)
        
        expected_columns = ['text', 'generated']
        self.assertTrue(all(col in df.columns for col in expected_columns), f"Expected columns {expected_columns}, but got {df.columns.tolist()}.")
        
        try:
            self.assertFalse(df[expected_columns].isnull().any().any(), "There are missing values in the 'text' or 'generated' columns.")
        except AssertionError as e:
            num_missing = df[expected_columns].isnull().sum()
            self.fail(f"{e} Number of missing values in 'text' column: {num_missing['text']}. Number of missing values in 'generated' column: {num_missing['generated']}.")
            raise e
        
        unique_values = df['generated'].unique()
        self.assertTrue(set(unique_values).issubset({0, 1}), f"The 'generated' column has unexpected values: {unique_values}. It should only contain 0 or 1.")

if __name__ == "__main__":
    unittest.main()
