import pandas as pd
from transformers import pipeline

df = pd.read_csv("test_1K_with_headers.csv")

pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

df['sentiment'] = df['review'].apply(lambda x: pipe(x)[0]['label'])
df['confidence'] = df['review'].apply(lambda x: pipe(x)[0]['score'])

df.to_csv("data_with_sentiment.csv", index=False)