import pandas as pd

df = pd.read_csv('data/preprocessed.csv', nrows=10)

df.to_csv('data/preprocessed.csv', index=False)
