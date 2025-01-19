import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from tqdm import tqdm

nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Preprocess text
def preprocess_text(text):
    text = text.lower()
    # Tokenize text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words])
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# Generate BERT embeddings
def bert_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in tqdm(texts, desc="Generating BERT embeddings"):
        inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)


# Load dataset
dataset_path = "/content/eng.csv"
data = load_data(dataset_path)

# Preprocess text
print("Preprocessing text...")
data["tokens"] = data["text"].apply(preprocess_text)
data["processed_text"] = data["tokens"].apply(lambda tokens: " ".join(tokens))

# Split data
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# TF-IDF Vectorization
print("Generating TF-IDF vectors...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit(train_data["processed_text"])
train_tfidf_vectors = tfidf_vectorizer.transform(train_data["processed_text"])
val_tfidf_vectors = tfidf_vectorizer.transform(val_data["processed_text"])

# Load DistilBERT
print("Loading DistilBERT model...")
bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Generate BERT embeddings
train_bert_vectors = bert_embeddings(
    train_data["processed_text"].tolist(), bert_tokenizer, bert_model
)
val_bert_vectors = bert_embeddings(
    val_data["processed_text"].tolist(), bert_tokenizer, bert_model
)

# Prepare labels
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(
    train_data[["Anger", "Fear", "Joy", "Sadness", "Surprise"]].values
)
val_labels = mlb.transform(
    val_data[["Anger", "Fear", "Joy", "Sadness", "Surprise"]].values
)

# Train Random Forest model on TF-IDF vectors
print("Training Random Forest model on TF-IDF vectors...")
rf_tfidf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_tfidf_model.fit(train_tfidf_vectors, train_labels)

# Train Random Forest model on BERT embeddings
print("Training Random Forest model on BERT embeddings...")
rf_bert_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_bert_model.fit(train_bert_vectors, train_labels)

mlp_bert_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
mlp_bert_model.fit(train_bert_vectors, train_labels)

mlp_tfidf_model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
mlp_tfidf_model.fit(train_tfidf_vectors, train_labels)


# Evaluate model
def evaluate_model(model, X_val, y_val, approach):
    y_pred = model.predict(X_val)
    print(f"\nEvaluation for {approach} approach:")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred, average="weighted"))
    print("Recall:", recall_score(y_val, y_pred, average="weighted"))
    print("F1-score:", f1_score(y_val, y_pred, average="weighted"))


# Evaluate
print("Evaluating models...")
evaluate_model(rf_tfidf_model, val_tfidf_vectors, val_labels, "RF - TF-IDF")
evaluate_model(rf_bert_model, val_bert_vectors, val_labels, "RF - BERT")
evaluate_model(mlp_bert_model, val_bert_vectors, val_labels, "MLP - BERT")
evaluate_model(mlp_tfidf_model, val_tfidf_vectors, val_labels, "MLP - TF-IDF")

