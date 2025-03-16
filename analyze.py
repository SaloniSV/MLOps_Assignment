from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

sentiment_pipeline = pipeline("sentiment-analysis")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


CLASS_FILE = "classes.json"  # Path to your classes JSON file

def load_classes(file_path=CLASS_FILE):
    try:
        with open(file_path, 'r') as file:
            # Load classes from the JSON file
            classes = json.load(file)
            return classes
    except FileNotFoundError:
        # If the file doesn't exist, return an empty list
        print(f"Warning: {file_path} not found. Returning empty class list.")
        return []
    except json.JSONDecodeError:
        # If there's a JSON decode error (e.g., corrupted file), return an empty list
        print(f"Error: {file_path} contains invalid JSON. Returning empty class list.")
        return []

def save_classes(classes, file_path=CLASS_FILE):
    try:
        with open(file_path, "w") as file:
            # Write the classes as JSON to the file
            json.dump(classes, file, indent=4)  # Pretty print with indent
    except Exception as e:
        print(f"Error saving classes: {e}")

# Load classes from the JSON file (initialization)
EMAIL_CLASSES = load_classes()


def get_sentiment(text):
    response = sentiment_pipeline(text)
    return response

def compute_embeddings(embeddings = EMAIL_CLASSES):
    embeddings = model.encode(embeddings)
    return zip(EMAIL_CLASSES, embeddings)

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
