#!/usr/bin/env python3
"""
Complete Training Pipeline for Sentiment Analysis Models
Trains TF-IDF+FFNN, E5 Embedding Classifier, and BERT with Optuna Optimization

Usage:
    # Train all models with default settings:
    python train_all_models.py
    
    # Train specific models only:
    python train_all_models.py --models tfidf bert
    
    # Skip Optuna optimization (use saved hyperparameters):
    python train_all_models.py --skip-optuna
    
    # Custom number of Optuna trials:
    python train_all_models.py --n-trials 5
    
    # Run in tmux for long training:
    tmux new -s training
    python train_all_models.py 2>&1 | tee training.log
"""

import os
import sys
import json
import pickle
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

import optuna
from optuna.samplers import TPESampler

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
MAX_LENGTH = 256
N_TRIALS_DEFAULT = 10
N_SPLITS = 3

# Paths
DATA_DIR = Path('../data/processed')
MODELS_DIR = Path('models')
LOGS_DIR = Path('outputs/logs')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

def load_json_data(filepath):
    """Load JSON dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_label_distribution(labels, name):
    """Print label distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} label distribution:")
    for label, count in zip(unique, counts):
        sentiment = 'Positive' if label == 1 else 'Negative'
        print(f"  {sentiment}: {count:,} ({count/len(labels)*100:.1f}%)")

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class FFNNClassifier(nn.Module):
    """Feed-Forward Neural Network for classification."""
    
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.3):
        super(FFNNClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EmbeddingClassifier(nn.Module):
    """Simple classifier head for pre-computed embeddings."""
    
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


class SentimentDataset(Dataset):
    """Dataset class for BERT tokenization."""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_ffnn(model, train_loader, val_loader, optimizer, criterion,
               num_epochs=20, patience=3, device='cpu', save_path=None):
    """Train FFNN with early stopping."""
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ New best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    return history, best_val_acc


def compute_metrics_bert(eval_pred):
    """Compute metrics for BERT evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# ============================================================================
# MODEL 1: TF-IDF + FFNN
# ============================================================================

def train_tfidf_ffnn(train_texts, train_labels, val_texts, val_labels,
                     device, skip_optuna=False, n_trials=15):
    """Train TF-IDF + FFNN model."""
    
    print_section("MODEL 1: TF-IDF + FEED-FORWARD NEURAL NETWORK")
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        lowercase=True,
        strip_accents='unicode',
        stop_words='english'
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts).toarray()
    X_val_tfidf = tfidf_vectorizer.transform(val_texts).toarray()
    
    print(f"✓ TF-IDF features created: {X_train_tfidf.shape}")
    
    # Save vectorizer
    (MODELS_DIR / 'tfidf_ffnn').mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / 'tfidf_ffnn' / 'tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    # Hyperparameter optimization with K-Fold CV
    if not skip_optuna:
        print("\nStarting K-Fold CV hyperparameter optimization...")
        print(f"Trials: {n_trials}, Folds: {N_SPLITS}")
        
        def objective_ffnn_kfold(trial):
            hidden_dim1 = trial.suggest_int('hidden_dim1', 128, 512, step=64)
            hidden_dim2 = trial.suggest_int('hidden_dim2', 64, 256, step=32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            print(f"\n[Trial {trial.number}] hidden={hidden_dim1},{hidden_dim2}, "
                  f"dropout={dropout_rate:.3f}, lr={learning_rate:.2e}, batch={batch_size}")
            
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            fold_scores = []
            
            X_combined = np.vstack([X_train_tfidf, X_val_tfidf])
            y_combined = np.array(train_labels + val_labels)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
                print(f"  Fold {fold_idx+1}/{N_SPLITS}...", end=" ")
                
                X_fold_train = X_combined[train_idx]
                y_fold_train = y_combined[train_idx]
                X_fold_val = X_combined[val_idx]
                y_fold_val = y_combined[val_idx]
                
                model = FFNNClassifier(
                    input_dim=X_train_tfidf.shape[1],
                    hidden_dims=[hidden_dim1, hidden_dim2],
                    dropout_rate=dropout_rate
                ).to(device)
                
                train_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_fold_train),
                    torch.LongTensor(y_fold_train)
                )
                val_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_fold_val),
                    torch.LongTensor(y_fold_val)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                _, best_val_acc = train_ffnn(
                    model, train_loader, val_loader, optimizer, criterion,
                    num_epochs=10, patience=2, device=device, save_path=None
                )
                
                fold_scores.append(best_val_acc)
                print(f"Acc={best_val_acc:.2f}%")
                
                del model, optimizer, train_loader, val_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            mean_acc = np.mean(fold_scores)
            print(f"  → Mean CV Accuracy: {mean_acc:.2f}%")
            return mean_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            study_name='tfidf_ffnn_kfold'
        )
        study.optimize(objective_ffnn_kfold, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✓ Optimization complete. Best CV accuracy: {study.best_value:.2f}%")
        best_params = study.best_params
        
        # Save study
        with open(MODELS_DIR / 'tfidf_ffnn' / 'optuna_study_kfold.pkl', 'wb') as f:
            pickle.dump(study, f)
    else:
        print("\n⏭️  Skipping Optuna optimization")
        hyperparams_file = MODELS_DIR / 'tfidf_ffnn' / 'hyperparameters.json'
        if hyperparams_file.exists():
            with open(hyperparams_file, 'r') as f:
                best_params = json.load(f)
            print("✓ Loaded saved hyperparameters")
        else:
            print("⚠️  No saved hyperparameters found, using defaults")
            best_params = {
                'hidden_dim1': 256,
                'hidden_dim2': 128,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64
            }
    
    # Train final model
    print("\n🎯 Training final TF-IDF + FFNN model...")
    final_model = FFNNClassifier(
        input_dim=X_train_tfidf.shape[1],
        hidden_dims=[best_params['hidden_dim1'], best_params['hidden_dim2']],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_tfidf),
        torch.LongTensor(train_labels)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_tfidf),
        torch.LongTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    history, best_val_acc = train_ffnn(
        final_model, train_loader, val_loader, optimizer, criterion,
        num_epochs=30, patience=5, device=device,
        save_path=MODELS_DIR / 'tfidf_ffnn' / 'best_model.pt'
    )
    
    # Save artifacts
    with open(MODELS_DIR / 'tfidf_ffnn' / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    with open(MODELS_DIR / 'tfidf_ffnn' / 'hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✅ TF-IDF + FFNN training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {MODELS_DIR / 'tfidf_ffnn'}")

# ============================================================================
# MODEL 2: E5 EMBEDDING CLASSIFIER
# ============================================================================

def train_e5_classifier(train_texts, train_labels, val_texts, val_labels,
                        device, skip_optuna=False, n_trials=12):
    """Train E5 embedding classifier."""
    
    print_section("MODEL 2: E5 EMBEDDING CLASSIFIER")
    
    # Load embedding model
    print("Loading E5 embedding model...")
    embedding_model_name = 'intfloat/e5-small-v2'
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    print(f"✓ Loaded: {embedding_model_name}")
    print(f"  Embedding dimension: {embedding_model.get_sentence_embedding_dimension()}")
    
    # Generate embeddings
    print("\nGenerating embeddings (this may take 5-10 minutes)...")
    train_texts_prefixed = [f"query: {text}" for text in train_texts]
    val_texts_prefixed = [f"query: {text}" for text in val_texts]
    
    X_train_embeddings = embedding_model.encode(
        train_texts_prefixed, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    X_val_embeddings = embedding_model.encode(
        val_texts_prefixed, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )
    
    print(f"✓ Embeddings generated: {X_train_embeddings.shape}")
    
    # Save embeddings and model info
    (MODELS_DIR / 'e5_classifier').mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / 'e5_classifier' / 'train_embeddings.npy', X_train_embeddings)
    np.save(MODELS_DIR / 'e5_classifier' / 'val_embeddings.npy', X_val_embeddings)
    with open(MODELS_DIR / 'e5_classifier' / 'embedding_model_name.txt', 'w') as f:
        f.write(embedding_model_name)
    
    # Hyperparameter optimization with K-Fold CV
    if not skip_optuna:
        print("\nStarting K-Fold CV hyperparameter optimization...")
        print(f"Trials: {n_trials}, Folds: {N_SPLITS}")
        
        def objective_embedding_kfold(trial):
            hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            
            print(f"\n[Trial {trial.number}] hidden={hidden_dim}, "
                  f"dropout={dropout_rate:.3f}, lr={learning_rate:.2e}, batch={batch_size}")
            
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            fold_scores = []
            
            X_combined = np.vstack([X_train_embeddings, X_val_embeddings])
            y_combined = np.array(train_labels + val_labels)
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
                print(f"  Fold {fold_idx+1}/{N_SPLITS}...", end=" ")
                
                X_fold_train = X_combined[train_idx]
                y_fold_train = y_combined[train_idx]
                X_fold_val = X_combined[val_idx]
                y_fold_val = y_combined[val_idx]
                
                model = EmbeddingClassifier(
                    embedding_dim=X_train_embeddings.shape[1],
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate
                ).to(device)
                
                train_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_fold_train),
                    torch.LongTensor(y_fold_train)
                )
                val_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_fold_val),
                    torch.LongTensor(y_fold_val)
                )
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                
                _, best_val_acc = train_ffnn(
                    model, train_loader, val_loader, optimizer, criterion,
                    num_epochs=10, patience=2, device=device, save_path=None
                )
                
                fold_scores.append(best_val_acc)
                print(f"Acc={best_val_acc:.2f}%")
                
                del model, optimizer, train_loader, val_loader
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            mean_acc = np.mean(fold_scores)
            print(f"  → Mean CV Accuracy: {mean_acc:.2f}%")
            return mean_acc
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            study_name='e5_classifier_kfold'
        )
        study.optimize(objective_embedding_kfold, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✓ Optimization complete. Best CV accuracy: {study.best_value:.2f}%")
        best_params = study.best_params
        
        # Save study
        with open(MODELS_DIR / 'e5_classifier' / 'optuna_study_kfold.pkl', 'wb') as f:
            pickle.dump(study, f)
    else:
        print("\n⏭️  Skipping Optuna optimization")
        hyperparams_file = MODELS_DIR / 'e5_classifier' / 'hyperparameters.json'
        if hyperparams_file.exists():
            with open(hyperparams_file, 'r') as f:
                best_params = json.load(f)
            print("✓ Loaded saved hyperparameters")
        else:
            print("⚠️  No saved hyperparameters found, using defaults")
            best_params = {
                'hidden_dim': 128,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64
            }
    
    # Train final model
    print("\n🎯 Training final E5 classifier...")
    final_model = EmbeddingClassifier(
        embedding_dim=X_train_embeddings.shape[1],
        hidden_dim=best_params['hidden_dim'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train_embeddings),
        torch.LongTensor(train_labels)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val_embeddings),
        torch.LongTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'])
    
    optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    history, best_val_acc = train_ffnn(
        final_model, train_loader, val_loader, optimizer, criterion,
        num_epochs=30, patience=5, device=device,
        save_path=MODELS_DIR / 'e5_classifier' / 'best_model.pt'
    )
    
    # Save artifacts
    with open(MODELS_DIR / 'e5_classifier' / 'training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    with open(MODELS_DIR / 'e5_classifier' / 'hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✅ E5 classifier training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"   Model saved to: {MODELS_DIR / 'e5_classifier'}")

# ============================================================================
# MODEL 3: BERT FINE-TUNING
# ============================================================================

def train_bert_model(train_texts, train_labels, val_texts, val_labels,
                     device, skip_optuna=False, n_trials=8):
    """Train BERT model with K-Fold CV optimization."""
    
    print_section("MODEL 3: BERT FINE-TUNING WITH K-FOLD CV")
    
    # Load tokenizer
    print("Loading BERT tokenizer and model...")
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Loaded: {model_name}")
    
    # Create datasets
    print("\nCreating tokenized datasets...")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    print(f"✓ Train dataset: {len(train_dataset):,} samples")
    print(f"✓ Val dataset: {len(val_dataset):,} samples")
    
    # Hyperparameter optimization with K-Fold CV
    if not skip_optuna:
        print("\nStarting K-Fold CV hyperparameter optimization for BERT...")
        print(f"Trials: {n_trials}, Folds: 2 (computational efficiency)")
        print("⚠️  This will take 1-3 hours depending on hardware\n")
        
        def objective_bert_kfold(trial):
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True)
            weight_decay = trial.suggest_float('weight_decay', 0.0, 0.1)
            warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.2)
            num_epochs = trial.suggest_int('num_epochs', 2, 4)
            batch_size = trial.suggest_categorical('batch_size', [8, 16])
            
            print(f"\n[Trial {trial.number}] lr={learning_rate:.2e}, wd={weight_decay:.3f}, "
                  f"warmup={warmup_ratio:.2f}, epochs={num_epochs}, batch={batch_size}")
            
            n_folds = 2
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
            fold_scores = []
            
            combined_texts = train_texts + val_texts
            combined_labels = train_labels + val_labels
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(combined_texts, combined_labels)):
                print(f"  Fold {fold_idx+1}/{n_folds}...", end=" ", flush=True)
                
                fold_train_texts = [combined_texts[i] for i in train_idx]
                fold_train_labels = [combined_labels[i] for i in train_idx]
                fold_val_texts = [combined_texts[i] for i in val_idx]
                fold_val_labels = [combined_labels[i] for i in val_idx]
                
                fold_train_dataset = SentimentDataset(
                    fold_train_texts, fold_train_labels, tokenizer, MAX_LENGTH
                )
                fold_val_dataset = SentimentDataset(
                    fold_val_texts, fold_val_labels, tokenizer, MAX_LENGTH
                )
                
                fold_model = AutoModelForSequenceClassification.from_pretrained(
                    'bert-base-uncased', num_labels=2,
                    output_attentions=False, output_hidden_states=False
                )
                
                fold_training_args = TrainingArguments(
                    output_dir=f'models/bert_finetuned/trial_{trial.number}_fold_{fold_idx}',
                    num_train_epochs=num_epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=32,
                    warmup_ratio=warmup_ratio,
                    weight_decay=weight_decay,
                    learning_rate=learning_rate,
                    logging_steps=200,
                    evaluation_strategy='epoch',
                    save_strategy='epoch',
                    save_total_limit=1,
                    load_best_model_at_end=True,
                    metric_for_best_model='f1',
                    greater_is_better=True,
                    fp16=torch.cuda.is_available(),
                    seed=SEED,
                    report_to='none',
                    logging_dir=None,
                    disable_tqdm=True
                )
                
                fold_trainer = Trainer(
                    model=fold_model,
                    args=fold_training_args,
                    train_dataset=fold_train_dataset,
                    eval_dataset=fold_val_dataset,
                    compute_metrics=compute_metrics_bert,
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
                )
                
                fold_trainer.train()
                eval_results = fold_trainer.evaluate()
                fold_f1 = eval_results['eval_f1']
                fold_scores.append(fold_f1)
                
                print(f"F1={fold_f1:.4f}", flush=True)
                
                del fold_model, fold_trainer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                if os.path.exists(f'models/bert_finetuned/trial_{trial.number}_fold_{fold_idx}'):
                    shutil.rmtree(f'models/bert_finetuned/trial_{trial.number}_fold_{fold_idx}')
            
            mean_f1 = np.mean(fold_scores)
            print(f"  → Mean CV F1: {mean_f1:.4f}")
            return mean_f1
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=SEED),
            study_name='bert_kfold'
        )
        study.optimize(objective_bert_kfold, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✓ Optimization complete. Best CV F1: {study.best_value:.4f}")
        best_params = study.best_params
        
        # Save study
        (MODELS_DIR / 'bert_finetuned').mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / 'bert_finetuned' / 'optuna_study_kfold.pkl', 'wb') as f:
            pickle.dump(study, f)
        with open(MODELS_DIR / 'bert_finetuned' / 'best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
    else:
        print("\n⏭️  Skipping Optuna optimization")
        hyperparams_file = MODELS_DIR / 'bert_finetuned' / 'best_hyperparameters.json'
        if hyperparams_file.exists():
            with open(hyperparams_file, 'r') as f:
                best_params = json.load(f)
            print("✓ Loaded saved hyperparameters")
        else:
            print("⚠️  No saved hyperparameters found, using defaults")
            best_params = {
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'warmup_ratio': 0.1,
                'num_epochs': 3,
                'batch_size': 16
            }
    
    # Train final BERT model
    print("\n🎯 Training final BERT model with best hyperparameters...")
    print(f"   Learning rate: {best_params['learning_rate']:.2e}")
    print(f"   Weight decay: {best_params['weight_decay']:.3f}")
    print(f"   Warmup ratio: {best_params['warmup_ratio']:.2f}")
    print(f"   Epochs: {best_params['num_epochs']}")
    print(f"   Batch size: {best_params['batch_size']}\n")
    
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2,
        output_attentions=False, output_hidden_states=False
    )
    
    training_args = TrainingArguments(
        output_dir='models/bert_finetuned',
        num_train_epochs=best_params['num_epochs'],
        per_device_train_batch_size=best_params['batch_size'],
        per_device_eval_batch_size=32,
        warmup_ratio=best_params['warmup_ratio'],
        weight_decay=best_params['weight_decay'],
        learning_rate=best_params['learning_rate'],
        logging_dir='outputs/logs/bert',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to='none'
    )
    
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_bert,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    train_result = trainer.train()
    
    # Save model
    trainer.save_model('models/bert_finetuned/final_model')
    tokenizer.save_pretrained('models/bert_finetuned/final_model')
    
    # Save metrics
    with open('models/bert_finetuned/training_metrics.json', 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    
    # Evaluate
    eval_results = trainer.evaluate()
    with open('models/bert_finetuned/validation_metrics.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✅ BERT training complete!")
    print(f"   Training loss: {train_result.metrics['train_loss']:.4f}")
    print(f"   Val F1 score: {eval_results['eval_f1']:.4f}")
    print(f"   Model saved to: {MODELS_DIR / 'bert_finetuned' / 'final_model'}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train sentiment analysis models (TF-IDF+FFNN, E5, BERT)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models with default settings:
  python train_all_models.py
  
  # Train only TF-IDF and BERT:
  python train_all_models.py --models tfidf bert
  
  # Skip Optuna optimization (use saved hyperparameters):
  python train_all_models.py --skip-optuna
  
  # Custom number of trials:
  python train_all_models.py --n-trials 5
  
  # Run in tmux:
  tmux new -s training
  python train_all_models.py 2>&1 | tee training.log
        """
    )
    
    parser.add_argument('--models', nargs='+', 
                        choices=['tfidf', 'e5', 'bert', 'all'],
                        default=['all'],
                        help='Models to train (default: all)')
    parser.add_argument('--skip-optuna', action='store_true',
                        help='Skip Optuna optimization, use saved hyperparameters')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Number of Optuna trials (default: 15 for TF-IDF, 12 for E5, 8 for BERT)')
    parser.add_argument('--data-dir', type=Path, default='../data/processed',
                        help='Data directory (default: ../data/processed)')
    parser.add_argument('--models-dir', type=Path, default='models',
                        help='Models output directory (default: models)')
    
    args = parser.parse_args()
    
    # Use paths from arguments
    data_dir = args.data_dir
    models_dir = args.models_dir
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = ['tfidf', 'e5', 'bert']
    else:
        models_to_train = args.models
    
    # Setup
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("SENTIMENT ANALYSIS MODEL TRAINING PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Seed: {SEED}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  K-fold splits: {N_SPLITS}")
    print(f"  Skip Optuna: {args.skip_optuna}")
    print(f"  Models to train: {', '.join(models_to_train)}")
    print(f"  Data directory: {data_dir}")
    print(f"  Models directory: {models_dir}")
    
    # Update global paths for functions to use
    global DATA_DIR, MODELS_DIR
    DATA_DIR = data_dir
    MODELS_DIR = models_dir
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print_section("LOADING DATA")
    print(f"Loading datasets from {DATA_DIR}...")
    
    train_data = load_json_data(DATA_DIR / 'train.json')
    val_data = load_json_data(DATA_DIR / 'eval.json')
    
    train_texts = [item['text'] for item in train_data]
    train_labels = [item['label'] for item in train_data]
    val_texts = [item['text'] for item in val_data]
    val_labels = [item['label'] for item in val_data]
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_texts):,} samples")
    print(f"  Val: {len(val_texts):,} samples")
    print_label_distribution(train_labels, "Training")
    print_label_distribution(val_labels, "Validation")
    
    # Train models
    start_time = pd.Timestamp.now()
    
    try:
        if 'tfidf' in models_to_train:
            n_trials = args.n_trials if args.n_trials else 15
            train_tfidf_ffnn(train_texts, train_labels, val_texts, val_labels,
                           device, args.skip_optuna, n_trials)
        
        if 'e5' in models_to_train:
            n_trials = args.n_trials if args.n_trials else 12
            train_e5_classifier(train_texts, train_labels, val_texts, val_labels,
                              device, args.skip_optuna, n_trials)
        
        if 'bert' in models_to_train:
            n_trials = args.n_trials if args.n_trials else 8
            train_bert_model(train_texts, train_labels, val_texts, val_labels,
                           device, args.skip_optuna, n_trials)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    end_time = pd.Timestamp.now()
    duration = (end_time - start_time).total_seconds() / 60
    
    print_section("✅ TRAINING COMPLETE!")
    print(f"Total training time: {duration:.1f} minutes")
    print(f"\nModels saved to: {MODELS_DIR.absolute()}")
    print("\nNext steps:")
    print("  1. Run first.ipynb if you haven't already (data preprocessing)")
    print("  2. Run third.ipynb for evaluation and comparison")
    print(f"\nTrained models: {', '.join(models_to_train)}")
    print("\nModel locations:")
    for model in models_to_train:
        if model == 'tfidf':
            print(f"  • TF-IDF + FFNN: {MODELS_DIR / 'tfidf_ffnn'}")
        elif model == 'e5':
            print(f"  • E5 Classifier: {MODELS_DIR / 'e5_classifier'}")
        elif model == 'bert':
            print(f"  • BERT: {MODELS_DIR / 'bert_finetuned' / 'final_model'}")
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
