from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

CLASS_FILE = "classes.json"  # Ensure this file exists

# Function to load classes from a JSON file
def load_classes():
    if os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, "r") as file:
            return json.load(file)
    else:
        return ["Work", "Sports", "Food"]  # Default classes if no file exists

# Function to save classes to a JSON file
def save_classes(classes):
    with open(CLASS_FILE, "w") as file:
        json.dump(classes, file)

def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings(embeddings=None):
    # Load classes if none are provided
    if embeddings is None:
        embeddings = load_classes()
    
    embeddings = model.encode(embeddings)
    return zip(embeddings, embeddings)  # Adjusted to return pairs of embeddings

def classify_email(text):
    # Encode the input text
    text_embedding = model.encode([text])[0]
    
    # Get embeddings for all classes
    class_embeddings = compute_embeddings()
    
    # Calculate distances and return results
    results = []
    for class_name, class_embedding in class_embeddings:
        # Compute cosine similarity between text and class embedding
        similarity = np.dot(text_embedding, class_embedding) / (np.linalg.norm(text_embedding) * np.linalg.norm(class_embedding))
        results.append({
            "class": class_name,
            "similarity": float(similarity)  # Convert tensor to float for JSON serialization
        })
    
    # Sort by similarity score descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results

