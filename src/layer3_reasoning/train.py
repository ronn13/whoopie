"""
=============================================================================
Layer 3: Text Classification Model Training
=============================================================================
Trains SafeAeroBERT on ADREP occurrence categories using labeled records
from the local PostgreSQL database (aviation.exp10_training_labels joined
with asn_scraped_accidents).

Codes with fewer than MIN_SAMPLES examples are folded into UNK.
=============================================================================
"""

import os
import json
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import pandas as pd
import psycopg2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "NASA-AIML/MIKA_SafeAeroBERT"
BATCH_SIZE   = 16
EPOCHS       = 8
LEARNING_RATE = 2e-5
MAX_LEN      = 256
MIN_SAMPLES  = 10   # codes with fewer examples → UNK
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "aviation",
    "user":     "postgres",
    "password": "toormaster",
}
# ─────────────────────────────────────────────────────────────────────────────


class AviationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_data():
    print("Connecting to database…")
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT a.narrative, e.primary_code
        FROM exp10_training_labels e
        JOIN asn_scraped_accidents a ON a.uid = e.event_id
        WHERE a.narrative IS NOT NULL AND a.narrative <> ''
    """, conn)
    conn.close()
    print(f"Loaded {len(df)} labeled records.")

    # Fold rare codes into UNK
    counts = df['primary_code'].value_counts()
    rare   = set(counts[counts < MIN_SAMPLES].index)
    if rare:
        print(f"Folding {len(rare)} rare codes into UNK: {sorted(rare)}")
    df['label_code'] = df['primary_code'].apply(lambda c: 'UNK' if c in rare else c)

    print("\nClass distribution after folding:")
    print(df['label_code'].value_counts().to_string())
    print()

    categories = sorted(df['label_code'].unique().tolist())
    cat2id = {c: i for i, c in enumerate(categories)}
    id2cat = {i: c for c, i in cat2id.items()}

    df['label'] = df['label_code'].map(cat2id)
    return df, cat2id, id2cat


def train():
    df, cat2id, id2cat = load_data()
    num_labels = len(cat2id)
    print(f"Training on {num_labels} classes: {sorted(cat2id.keys())}\n")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['narrative'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label'].tolist(),
    )

    print("Loading tokenizer and model…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model     = model.to(DEVICE)

    train_dataset = AviationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset   = AviationDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # Class-weighted loss to counter imbalance
    weights  = compute_class_weight('balanced', classes=np.arange(num_labels), y=train_labels)
    loss_fn  = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE))

    print("Starting training…\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            loss           = loss_fn(outputs.logits, labels)
            total_loss    += loss.item()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} avg loss: {total_loss/len(train_loader):.4f}")

    print("\nEvaluating on validation set…")
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            outputs = model(
                input_ids=batch['input_ids'].to(DEVICE),
                attention_mask=batch['attention_mask'].to(DEVICE),
            )
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(batch['labels'].tolist())

    target_names = [id2cat[i] for i in range(num_labels)]
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names))

    # Save weights + id2cat
    os.makedirs(os.path.join(ROOT, "outputs", "models"), exist_ok=True)
    weights_path = os.path.join(ROOT, "outputs", "models", "safeaerobert_classifier.pt")
    mapping_path = os.path.join(ROOT, "outputs", "models", "id2cat.json")
    torch.save(model.state_dict(), weights_path)
    with open(mapping_path, "w") as f:
        json.dump(id2cat, f, indent=2)
    print(f"\nWeights → {weights_path}")
    print(f"Mapping → {mapping_path}")


if __name__ == "__main__":
    train()
