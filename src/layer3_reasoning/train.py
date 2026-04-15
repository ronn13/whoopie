"""
=============================================================================
Layer 3: Text Classification Model Training
=============================================================================
This script is the "Layer 3" architecture of the machine learning pipeline.

What it does:
1. Self-fetching Data: Connects to the database and pulls rows possessing a 
   valid 'narrative' and a recorded ADREP 'category' (e.g., 'Loss of control').
2. Text Encoding: Maps every unique category to an integer (a multi-class 
   prediction task) and splits the dataset into 80% training / 20% validation.
3. Fine-Tuning a Neural Network: Takes the NASA-AIML/MIKA_SafeAeroBERT text
   embedding model and attaches a PyTorch linear classification head.
4. Learning DB Patterns: Over several epochs, it feeds narratives in batches.
   The BERT model learns textual patterns and keywords that lead to specific
   categories and adjusts its gradient weights (AdamW).
5. Taking a Final Exam: Predicts the category for the remaining 20% unseen 
   data and prints a detailed scikit-learn scorecard computing Precision/F1.

Output:
Saves the trained weights to 'outputs/models/safeaerobert_classifier.pt'.
=============================================================================
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import sys

# Ensure the tests folder is directly in the path since it is not a proper module package
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from db_connect import get_connection

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "NASA-AIML/MIKA_SafeAeroBERT"
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

class AviationWarningDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

import json

def load_data():
    print("Fetching data from labeled hard cases CSV...")
    df = pd.read_csv(os.path.join(ROOT, "data", "reference", "labeled_hard_cases.csv"))
    
    # Ensure no empty category
    df = df[df['adrep_category'].notna()]
    df = df[df['narrative_1'].notna()]
    
    print(f"Fetched {len(df)} records with categories.")
    
    # Map string categories to integers
    categories = sorted(df['adrep_category'].unique().tolist())
    cat2id = {c: i for i, c in enumerate(categories)}
    id2cat = {i: c for c, i in cat2id.items()}
    
    df['label'] = df['adrep_category'].map(cat2id)
    df['narrative'] = df['narrative_1']
    
    return df, cat2id, id2cat

def train():
    df, cat2id, id2cat = load_data()
    num_labels = len(cat2id)
    print(f"Total unique ADREP categories: {num_labels}")

    # Train/Test Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['narrative'].tolist(), 
        df['label'].tolist(), 
        test_size=0.2, 
        random_state=42
    )

    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model = model.to(DEVICE)

    train_dataset = AviationWarningDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = AviationWarningDataset(val_texts, val_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting Training Loop...\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss/len(train_loader):.4f}")

    print("\nEvaluating Model on Validation Set...")
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    target_names = [id2cat[i] for i in range(num_labels)]
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names))

    # Save the fine-tuned model
    os.makedirs(os.path.join(ROOT, "outputs", "models"), exist_ok=True)
    save_path = os.path.join(ROOT, "outputs", "models", "safeaerobert_classifier.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")
    
    # Save the id2cat mapping for inference
    mapping_path = os.path.join(ROOT, "outputs", "models", "id2cat.json")
    with open(mapping_path, "w") as f:
        json.dump(id2cat, f)
    print(f"Label mapping saved to {mapping_path}")

if __name__ == "__main__":
    train()
