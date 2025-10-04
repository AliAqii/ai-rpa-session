#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arabic Text Classification - Complete Fixed Version
====================================================
- Uses enriched dataset with 100+ samples
- Includes Jordanian dialect and slang
- Returns clean Python strings (not np.str_)
- Better accuracy with expanded vocabulary
- Ready for production use at Zain

Author: AI Assistant for Zain Data Engineering
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import re
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CSV_FILE = 'arabic_sentiment_enriched.csv'  # Your CSV file path
TEST_SIZE = 0.2  # 20% for testing
RANDOM_STATE = 42

# ==============================================================================
# PREPROCESSING FUNCTIONS
# ==============================================================================

def preprocess_arabic_text(text):
    """
    Clean and normalize Arabic text

    Steps:
    1. Remove Arabic diacritics (tashkeel)
    2. Normalize Arabic letters (hamza variations, ta marbuta)
    3. Remove punctuation and special characters
    4. Remove extra whitespace

    Args:
        text (str): Raw Arabic text

    Returns:
        str: Cleaned and normalized text
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove Arabic diacritics (تشكيل)
    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid (Shadda)
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(arabic_diacritics, '', text)

    # Normalize Arabic letters
    text = re.sub("[إأآا]", "ا", text)  # Normalize Alef variations
    text = re.sub("ى", "ي", text)        # Normalize Ya
    text = re.sub("ؤ", "ء", text)        # Normalize Hamza on Waw
    text = re.sub("ئ", "ء", text)        # Normalize Hamza on Ya
    text = re.sub("ة", "ه", text)        # Normalize Ta Marbuta

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def load_and_prepare_data(csv_file):
    """
    Load data from CSV and preprocess

    Args:
        csv_file (str): Path to CSV file with 'text' and 'label' columns

    Returns:
        tuple: (X, y) - cleaned texts and labels
    """
    print("="*70)
    print("LOADING AND PREPROCESSING DATA")
    print("="*70)

    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"\n✓ Loaded {len(df)} records from: {csv_file}")
    except Exception as e:
        print(f"\n✗ Error loading file: {str(e)}")
        return None, None

    # Validate columns
    if 'text' not in df.columns or 'label' not in df.columns:
        print("\n✗ Error: CSV must have 'text' and 'label' columns")
        return None, None

    # Clean data
    df = df.dropna(subset=['text', 'label'])
    print(f"\nPreprocessing {len(df)} samples...")
    df['cleaned_text'] = df['text'].apply(preprocess_arabic_text)
    df = df[df['cleaned_text'].str.len() > 0]

    print(f"✓ Preprocessing complete")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())

    return df['cleaned_text'], df['label']


def train_models(X, y):
    """
    Train multiple classification models and select the best

    Args:
        X: Text features
        y: Labels

    Returns:
        tuple: (tfidf_vectorizer, best_model, model_name, all_results)
    """
    print("\n" + "="*70)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*70)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing:  {len(X_test)} samples")

    # TF-IDF Feature Extraction
    print(f"\nExtracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=500,       # Increased from 200
        ngram_range=(1, 3),     # Unigrams, bigrams, and trigrams
        min_df=1,               # Minimum document frequency
        sublinear_tf=True,      # Apply sublinear scaling
        analyzer='word'         # Word-level analysis
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"✓ Feature extraction complete")
    print(f"  Vocabulary size: {len(tfidf.get_feature_names_out())}")
    print(f"  Feature matrix shape: {X_train_tfidf.shape}")

    # Train multiple models
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            C=1.0,
            random_state=RANDOM_STATE
        ),
        'SVM': SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE
        )
    }

    results = {}
    best_accuracy = 0
    best_model_name = None

    for name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"Training: {name}")
        print(f"{'-'*70}")

        # Train
        model.fit(X_train_tfidf, y_train)

        # Predict
        y_pred = model.predict(X_test_tfidf)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred
        }

        print(f"\nAccuracy: {accuracy:.2%}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.2%}")
    print(f"{'='*70}")

    return tfidf, results[best_model_name]['model'], best_model_name, results


# ==============================================================================
# PREDICTION FUNCTIONS
# ==============================================================================

def predict_sentiment(text, tfidf_vectorizer, model):
    """
    Predict sentiment for a single Arabic text

    Args:
        text (str): Input Arabic text
        tfidf_vectorizer: Trained TF-IDF vectorizer
        model: Trained classification model

    Returns:
        tuple: (prediction: str, confidence: float)
    """
    # Clean text
    cleaned = preprocess_arabic_text(text)

    # Transform to features
    features = tfidf_vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(features)

    # Convert numpy type to Python string
    prediction_str = str(prediction[0])

    # Get confidence if available
    try:
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        return prediction_str, confidence
    except:
        return prediction_str, None


def predict_batch(texts, tfidf_vectorizer, model):
    """
    Predict sentiment for multiple texts

    Args:
        texts (list): List of Arabic texts
        tfidf_vectorizer: Trained TF-IDF vectorizer
        model: Trained classification model

    Returns:
        list: List of (text, prediction, confidence) tuples
    """
    results = []

    for text in texts:
        pred, conf = predict_sentiment(text, tfidf_vectorizer, model)
        results.append((text, pred, conf))

    return results


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*70)
    print("ARABIC SENTIMENT ANALYSIS - ZAIN")
    print("="*70)

    # Load and prepare data
    X, y = load_and_prepare_data(CSV_FILE)

    if X is None or y is None:
        print("\n✗ Failed to load data. Exiting.")
        return

    # Train models
    tfidf, best_model, model_name, all_results = train_models(X, y)

    # Test with examples
    print("\n" + "="*70)
    print("TESTING WITH NEW EXAMPLES")
    print("="*70)

    test_examples = [
        "الخدمة رائعة والموظفين محترمين",
        "التطبيق زفت وما بيشتغل",
        "تجربة ممتازة وسهلة الاستخدام",
        "خدمة العملاء بطيئة جداً ومحبطة",
        "النت سريع والتغطية ممتازة",
        "الفاتورة غالية وفيها مصاريف مخفية",
        "الموظف كتير متعاون ومحترم",
        "الخدمة سيئة وما بتستاهل"
    ]

    for text in test_examples:
        pred, conf = predict_sentiment(text, tfidf, best_model)

        if conf:
            print(f"\nText: {text}")
            print(f"→ {pred} (Confidence: {conf:.1%})")
        else:
            print(f"\nText: {text}")
            print(f"→ {pred}")

    # Save model (optional)
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    try:
        import joblib
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        joblib.dump(best_model, 'sentiment_model.pkl')

        # Save model info
        with open('model_info.txt', 'w', encoding='utf-8') as f:
            f.write(f"Model Type: {model_name}\n")
            f.write(f"Accuracy: {all_results[model_name]['accuracy']:.2%}\n")
            f.write(f"Vocabulary Size: {len(tfidf.get_feature_names_out())}\n")
            f.write(f"Training Date: {pd.Timestamp.now()}\n")

        print("\n✓ Model saved successfully!")
        print("  - tfidf_vectorizer.pkl")
        print("  - sentiment_model.pkl")
        print("  - model_info.txt")

        print("\nTo load the model later:")
        print("  import joblib")
        print("  tfidf = joblib.load('tfidf_vectorizer.pkl')")
        print("  model = joblib.load('sentiment_model.pkl')")

    except Exception as e:
        print(f"\n⚠ Could not save model: {str(e)}")

    print("\n" + "="*70)
    print("✓ MODEL READY FOR PRODUCTION!")
    print("="*70)

    return tfidf, best_model, model_name


if __name__ == "__main__":
    tfidf, model, model_name = main()