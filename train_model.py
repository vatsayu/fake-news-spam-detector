# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# 1. Load the Kaggle Dataset
print("Loading Kaggle Fake News Dataset...")
# Replace 'train.csv' with your actual filename if it's named differently
df = pd.read_csv("WELFake_Dataset.csv")

# 2. Data Cleaning & Preprocessing
print("Cleaning data and handling missing values...")
# Drop rows where critical columns are empty
df = df.dropna(subset=['text', 'label'])

# Ensure labels are integers
df['label'] = df['label'].astype(int)

# Optional: If your dataset text is too clean or short, you can combine Title + Text
# df['text'] = df['title'].fillna('') + " " + df['text']

print(f"Dataset Loaded Successfully! Total training samples: {len(df)}")
print(f"Class Distribution:\n{df['label'].value_counts()}")

# 3. Split Data into Train and Test Sets (80% train, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label'] # Ensures equal distribution of fake/real news in both sets
)

# 4. Build the Pipeline: Vectorizer -> Classifier
# max_features limits the vocabulary size to keep memory usage low on local machines
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, max_features=10000)),
    ('nb', MultinomialNB())
])

# 5. Train Model
print("\nTraining Naive Bayes classifier on Kaggle data (this might take a minute)...")
pipeline.fit(X_train, y_train)

# 6. Evaluate Accuracy
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {test_acc * 100:.2f}%")

# 7. Save the trained pipeline to disk
with open('naive_bayes_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\nModel successfully saved as 'naive_bayes_model.pkl'!")