import pandas as pd
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def remove_leakage(text):
    """
    Advanced cleaning to remove data leakage sources.
    Removes agency names like 'Reuters' which cause the model to cheat.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    
    # 1. Remove specific leakage words (agency names, common leakage terms)
    # \b ensures we match whole words only
    text = re.sub(r'\b(?:reuters|wire|image|via|ap|fp|afp)\b', '', text)
    
    # 2. Remove anything inside parentheses, e.g., (Reuters)
    text = re.sub(r'\(.*?\)', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Remove extra punctuation and numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 5. Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_features(data_path, save_path='models/tfidf_vectorizer.joblib'):
    """
    Loads data, removes leakage/duplicates, and creates TF-IDF features.
    """
    print("Loading and processing data...")
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Ensure correct column name handling
    text_col = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
    print(f"Using column: {text_col}")
    
    # Handling missing values
    df[text_col] = df[text_col].fillna('')
    
    # --- CRITICAL FIX: DATA LEAKAGE PREVENTION ---
    print(f"Original shape: {df.shape}")
    
    # 1. Drop Duplicates (Prevents memorization)
    df.drop_duplicates(subset=[text_col], inplace=True)
    print(f"Shape after dropping duplicates: {df.shape}")
    
    # 2. Apply Leakage Removal (Removes 'Reuters', etc.)
    print("Applying leakage removal...")
    df[text_col] = df[text_col].apply(remove_leakage)
    
    # Remove empty rows created after cleaning
    df = df[df[text_col].str.strip().astype(bool)]
    
    # --- FEATURE ENGINEERING ---
    
    # Initialize TF-IDF vectorizer
    # max_features=5000 is a good balance for performance
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    print("Vectorizing text...")
    X = tfidf.fit_transform(df[text_col])
    y = df['label'] # Ensure your target column is named 'label'
    
    # Create models directory if not exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save vectorizer for later inference (MLOps principle)
    joblib.dump(tfidf, save_path)
    print(f"Vectorizer saved at: {save_path}")
    print(f"Final Feature Matrix: {X.shape}")
    
    return X, y

if __name__ == "__main__":
    # Quick test
    # Ensure the path matches your project structure
    try:
        X, y = create_features('data/processed/cleaned_data.csv')
        print("Feature creation successful.")
    except Exception as e:
        print(f"Error: {e}")