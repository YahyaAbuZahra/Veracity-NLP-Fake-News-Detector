import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from pathlib import Path

# Project paths (Clean & Stable)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "full_dataset.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_data.csv"

print(" Project root:", BASE_DIR)
print(" Input file:", INPUT_FILE)

# NLTK setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    clean_words = [w for w in words if w not in stop_words]
    return " ".join(clean_words)

# Data Processing

def process_dataframe(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['text'])
    df['cleaned_text'] = df['text'].apply(clean_text)
    df[['cleaned_text', 'label']].to_csv(output_path, index=False)
    print(" Cleaned data saved to:", output_path)


# Main

if __name__ == "__main__":
 
    test_text = "Hello! This is a sample News article, check it out at https://google.com"
    print("Original:", test_text)
    print("Cleaned :", clean_text(test_text))

    process_dataframe(INPUT_FILE, OUTPUT_FILE)
