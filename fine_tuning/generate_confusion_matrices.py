#!/usr/bin/env python3
"""
Standalone script to generate confusion matrices comparing in-domain vs cross-domain performance.
This script loads all trained models and evaluates them to create the visualization.

Usage:
    python generate_confusion_matrices.py

Requirements:
    - Trained models in models/ directory
    - Test data in ../data/processed/
    - Virtual environment with required packages activated
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

from sklearn.metrics import confusion_matrix, accuracy_score

# Configuration
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
DATA_DIR = Path('../data/processed')
MODELS_DIR = Path('models')
OUTPUT_DIR = Path('outputs/comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Model Classes
# ============================================================================

class FFNNClassifier(nn.Module):
    """Feed-Forward Neural Network for classification"""
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(FFNNClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EmbeddingClassifier(nn.Module):
    """Simple classifier head for pre-computed embeddings"""
    def __init__(self, embedding_dim, hidden_dim, dropout_rate=0.3):
        super(EmbeddingClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.classifier(x)


# ============================================================================
# Data Loading
# ============================================================================

def load_json_data(filepath):
    """Load JSON dataset"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

imdb_eval = load_json_data(DATA_DIR / 'imdb_eval.json')
yelp_eval = load_json_data(DATA_DIR / 'yelp_eval.json')
amazon_eval = load_json_data(DATA_DIR / 'amazon_eval.json')

imdb_texts = [item['text'] for item in imdb_eval]
imdb_labels = [item['label'] for item in imdb_eval]

yelp_texts = [item['text'] for item in yelp_eval]
yelp_labels = [item['label'] for item in yelp_eval]

amazon_texts = [item['text'] for item in amazon_eval]
amazon_labels = [item['label'] for item in amazon_eval]

print(f"✓ Data loaded:")
print(f"  IMDB:   {len(imdb_texts):,} samples")
print(f"  Yelp:   {len(yelp_texts):,} samples")
print(f"  Amazon: {len(amazon_texts):,} samples")


# ============================================================================
# Helper Functions
# ============================================================================

def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate PyTorch model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def predict_bert(texts, labels, model, tokenizer, batch_size=32, device='cpu'):
    """Generate predictions with BERT model"""
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
            batch_texts = texts[i:i+batch_size]
            encoding = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            all_preds.extend(predicted.cpu().numpy())
    
    return np.array(all_preds), np.array(labels)


# ============================================================================
# Load and Evaluate Models
# ============================================================================

results = {}

# ----------------------------------------------------------------------------
# 1. TF-IDF + FFNN
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("1. LOADING TF-IDF + FFNN MODEL")
print("="*70)

try:
    with open(MODELS_DIR / 'tfidf_ffnn' / 'tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    
    with open(MODELS_DIR / 'tfidf_ffnn' / 'hyperparameters.json', 'r') as f:
        tfidf_params = json.load(f)
    
    tfidf_model = FFNNClassifier(
        input_dim=tfidf_vectorizer.max_features,
        hidden_dims=[tfidf_params['hidden_dim1'], tfidf_params['hidden_dim2']],
        dropout_rate=tfidf_params['dropout_rate']
    ).to(device)
    
    tfidf_model.load_state_dict(torch.load(MODELS_DIR / 'tfidf_ffnn' / 'best_model.pt', map_location=device))
    tfidf_model.eval()
    
    print("✓ TF-IDF model loaded")
    
    # Transform data
    print("Transforming data with TF-IDF...")
    X_imdb_tfidf = tfidf_vectorizer.transform(imdb_texts).toarray()
    X_yelp_tfidf = tfidf_vectorizer.transform(yelp_texts).toarray()
    X_amazon_tfidf = tfidf_vectorizer.transform(amazon_texts).toarray()
    
    # Evaluate
    print("Evaluating TF-IDF model...")
    batch_size = 128
    
    imdb_loader = DataLoader(TensorDataset(torch.FloatTensor(X_imdb_tfidf), torch.LongTensor(imdb_labels)), batch_size=batch_size)
    yelp_loader = DataLoader(TensorDataset(torch.FloatTensor(X_yelp_tfidf), torch.LongTensor(yelp_labels)), batch_size=batch_size)
    amazon_loader = DataLoader(TensorDataset(torch.FloatTensor(X_amazon_tfidf), torch.LongTensor(amazon_labels)), batch_size=batch_size)
    
    imdb_preds, imdb_true = evaluate_model(tfidf_model, imdb_loader, device)
    yelp_preds, yelp_true = evaluate_model(tfidf_model, yelp_loader, device)
    amazon_preds, amazon_true = evaluate_model(tfidf_model, amazon_loader, device)
    
    results['TF-IDF + FFNN'] = {
        'imdb': (imdb_preds, imdb_true),
        'yelp': (yelp_preds, yelp_true),
        'amazon': (amazon_preds, amazon_true)
    }
    print("✓ TF-IDF evaluation complete")
    
except Exception as e:
    print(f"⚠️  TF-IDF model failed: {e}")
    results['TF-IDF + FFNN'] = None


# ----------------------------------------------------------------------------
# 2. E5 Classifier
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("2. LOADING E5 CLASSIFIER")
print("="*70)

try:
    model_path = MODELS_DIR / 'e5_classifier' / 'best_model.pt'
    if not model_path.exists():
        raise FileNotFoundError("E5 model file not found")
    
    with open(MODELS_DIR / 'e5_classifier' / 'embedding_model_name.txt', 'r') as f:
        embedding_model_name = f.read().strip()
    
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    
    with open(MODELS_DIR / 'e5_classifier' / 'hyperparameters.json', 'r') as f:
        e5_params = json.load(f)
    
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    e5_classifier = EmbeddingClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=e5_params['hidden_dim'],
        dropout_rate=e5_params['dropout_rate']
    ).to(device)
    
    e5_classifier.load_state_dict(torch.load(model_path, map_location=device))
    e5_classifier.eval()
    
    print("✓ E5 model loaded")
    
    # Generate embeddings
    print("Generating embeddings...")
    if 'e5' in embedding_model_name.lower():
        imdb_texts_prefixed = [f"query: {text}" for text in imdb_texts]
        yelp_texts_prefixed = [f"query: {text}" for text in yelp_texts]
        amazon_texts_prefixed = [f"query: {text}" for text in amazon_texts]
    else:
        imdb_texts_prefixed = imdb_texts
        yelp_texts_prefixed = yelp_texts
        amazon_texts_prefixed = amazon_texts
    
    X_imdb_embed = embedding_model.encode(imdb_texts_prefixed, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    X_yelp_embed = embedding_model.encode(yelp_texts_prefixed, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    X_amazon_embed = embedding_model.encode(amazon_texts_prefixed, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    # Evaluate
    print("Evaluating E5 model...")
    batch_size = 128
    
    imdb_loader = DataLoader(TensorDataset(torch.FloatTensor(X_imdb_embed), torch.LongTensor(imdb_labels)), batch_size=batch_size)
    yelp_loader = DataLoader(TensorDataset(torch.FloatTensor(X_yelp_embed), torch.LongTensor(yelp_labels)), batch_size=batch_size)
    amazon_loader = DataLoader(TensorDataset(torch.FloatTensor(X_amazon_embed), torch.LongTensor(amazon_labels)), batch_size=batch_size)
    
    imdb_preds, imdb_true = evaluate_model(e5_classifier, imdb_loader, device)
    yelp_preds, yelp_true = evaluate_model(e5_classifier, yelp_loader, device)
    amazon_preds, amazon_true = evaluate_model(e5_classifier, amazon_loader, device)
    
    results['E5 Classifier'] = {
        'imdb': (imdb_preds, imdb_true),
        'yelp': (yelp_preds, yelp_true),
        'amazon': (amazon_preds, amazon_true)
    }
    print("✓ E5 evaluation complete")
    
except Exception as e:
    print(f"⚠️  E5 model failed: {e}")
    results['E5 Classifier'] = None


# ----------------------------------------------------------------------------
# 3. BERT Fine-tuned
# ----------------------------------------------------------------------------
print("\n" + "="*70)
print("3. LOADING BERT MODEL")
print("="*70)

try:
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        MODELS_DIR / 'bert_finetuned' / 'final_model'
    ).to(device)
    bert_tokenizer = AutoTokenizer.from_pretrained(MODELS_DIR / 'bert_finetuned' / 'final_model')
    bert_model.eval()
    
    print("✓ BERT model loaded")
    
    # Evaluate
    print("Evaluating BERT model...")
    batch_size = 32 if torch.cuda.is_available() else 16
    
    imdb_preds, imdb_true = predict_bert(imdb_texts, imdb_labels, bert_model, bert_tokenizer, batch_size, device)
    yelp_preds, yelp_true = predict_bert(yelp_texts, yelp_labels, bert_model, bert_tokenizer, batch_size, device)
    amazon_preds, amazon_true = predict_bert(amazon_texts, amazon_labels, bert_model, bert_tokenizer, batch_size, device)
    
    results['Fine-tuned BERT'] = {
        'imdb': (imdb_preds, imdb_true),
        'yelp': (yelp_preds, yelp_true),
        'amazon': (amazon_preds, amazon_true)
    }
    print("✓ BERT evaluation complete")
    
except Exception as e:
    print(f"⚠️  BERT model failed: {e}")
    results['Fine-tuned BERT'] = None


# ============================================================================
# Generate Confusion Matrices
# ============================================================================

print("\n" + "="*70)
print("GENERATING CONFUSION MATRICES")
print("="*70)

# Filter out failed models
model_names = [name for name, res in results.items() if res is not None]
results_list = [results[name] for name in model_names]

if len(model_names) == 0:
    print("❌ No models loaded successfully. Cannot generate confusion matrices.")
    exit(1)

print(f"Models to plot: {model_names}")

# Create figure
if len(model_names) == 2:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
elif len(model_names) == 3:
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
else:
    fig, axes = plt.subplots(len(model_names), 2, figsize=(14, 6*len(model_names)))
    if len(model_names) == 1:
        axes = axes.reshape(1, 2)

for idx, (model_name, model_results) in enumerate(zip(model_names, results_list)):
    # In-Domain (IMDB + Yelp combined)
    ax_in = axes[idx, 0]
    
    imdb_preds, imdb_true = model_results['imdb']
    yelp_preds, yelp_true = model_results['yelp']
    
    in_domain_preds = np.concatenate([imdb_preds, yelp_preds])
    in_domain_true = np.concatenate([imdb_true, yelp_true])
    
    cm_in = confusion_matrix(in_domain_true, in_domain_preds)
    cm_in_normalized = cm_in.astype('float') / cm_in.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_in_normalized, annot=True, fmt='.3f', cmap='Blues', 
                ax=ax_in, cbar_kws={'label': 'Proportion'},
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    acc_in = accuracy_score(in_domain_true, in_domain_preds)
    ax_in.set_title(f'{model_name}\nIn-Domain (IMDB + Yelp)', 
                   fontsize=11, fontweight='bold')
    ax_in.set_ylabel('True Label', fontsize=10)
    ax_in.set_xlabel('Predicted Label', fontsize=10)
    
    # Cross-Domain (Amazon)
    ax_cross = axes[idx, 1]
    
    amazon_preds, amazon_true = model_results['amazon']
    cm_cross = confusion_matrix(amazon_true, amazon_preds)
    cm_cross_normalized = cm_cross.astype('float') / cm_cross.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_cross_normalized, annot=True, fmt='.3f', cmap='Oranges', 
                ax=ax_cross, cbar_kws={'label': 'Proportion'},
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    
    acc_cross = accuracy_score(amazon_true, amazon_preds)
    ax_cross.set_title(f'{model_name}\nCross-Domain (Amazon)', 
                      fontsize=11, fontweight='bold')
    ax_cross.set_ylabel('True Label', fontsize=10)
    ax_cross.set_xlabel('Predicted Label', fontsize=10)

fig.suptitle('Confusion Matrices: In-Domain vs Cross-Domain Performance', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_file = OUTPUT_DIR / 'confusion_matrices_in_vs_cross_domain.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")

# Display if running interactively
try:
    plt.show()
except:
    pass

print("\n" + "="*70)
print("✅ CONFUSION MATRICES GENERATED SUCCESSFULLY")
print("="*70)
print(f"\nOutput file: {output_file.absolute()}")
print(f"Models included: {', '.join(model_names)}")
