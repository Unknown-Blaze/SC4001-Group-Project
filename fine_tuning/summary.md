# **Comprehensive Code Summary: Traditional Fine-tuning vs LoRA for Cross-Domain Sentiment Analysis**

## **1. Research Question & Methodology**

This project compares two approaches for sentiment analysis across domains (IMDB, Yelp, Amazon):
- **Method A**: Traditional fine-tuning (TF-IDF+FFNN, E5 classifier, BERT)
- **Method B**: LoRA-based parameter-efficient fine-tuning with domain adaptation

**Key Design Principles:**
- **Fair Comparison**: Both methods use identical datasets, seed=42, max_length=256, eval_ratio=0.2
- **Dataset Size**: 30k samples per domain (IMDB, Yelp) for training
- **Hyperparameter Optimization**: Optuna with n_trials=10, n_splits=3 K-fold CV
- **Evaluation**: In-domain performance (IMDB+Yelp) vs cross-domain transfer (Amazon)

---

## **2. Data Preprocessing (Shared Component)**

**File**: `fine_tuning/first.ipynb`

### **2.1 Dataset Loading**
Uses HuggingFace `datasets` library:
```python
from datasets import load_dataset

imdb = load_dataset('imdb')
yelp = load_dataset('yelp_polarity')
amazon = load_dataset('amazon_polarity')
```

### **2.2 Sampling Strategy**
Three specialized functions ensure consistent data across methods:

**IMDB Special Handling** (matches LoRA exactly):
```python
def build_imdb_30k(imdb_dataset, seed=42):
    # Uses ALL 25k training samples + 5k from test
    train_data = imdb_dataset['train']  # All 25k
    test_data = imdb_dataset['test'].shuffle(seed=seed).select(range(5000))
    combined = concatenate_datasets([train_data, test_data])
    return combined.shuffle(seed=seed)
```

**Yelp & Amazon** (standard sampling):
```python
def build_yelp_30k(yelp_dataset, seed=42):
    # Sample 15k positive + 15k negative from training set
    train_data = yelp_dataset['train']
    pos = train_data.filter(lambda x: x['label'] == 1).shuffle(seed=seed).select(range(15000))
    neg = train_data.filter(lambda x: x['label'] == 0).shuffle(seed=seed).select(range(15000))
    combined = concatenate_datasets([pos, neg])
    return combined.shuffle(seed=seed)
```

### **2.3 Train/Eval Split**
```python
SEED = 42
EVAL_RATIO = 0.2

# Split each 30k dataset into 24k train + 6k eval
imdb_split = imdb_30k.train_test_split(test_size=EVAL_RATIO, seed=SEED)
yelp_split = yelp_30k.train_test_split(test_size=EVAL_RATIO, seed=SEED)
amazon_split = amazon_30k.train_test_split(test_size=EVAL_RATIO, seed=SEED)
```

### **2.4 Output Files**
Saves to `../data/processed/`:
- **Combined datasets**: `train.json` (48k), `eval.json` (12k), `amazon_test.json` (6k)
- **Separate datasets** (for individual evaluation):
  - `imdb_train.json`, `imdb_eval.json`
  - `yelp_train.json`, `yelp_eval.json`
  - `amazon_train.json`, `amazon_eval.json`

---

## **3. Method A - Traditional Fine-tuning**

**File**: `fine_tuning/second.ipynb`

### **3.1 Three Approaches**

#### **Baseline: TF-IDF + FFNN**
- TF-IDF vectorization (scikit-learn)
- Feed-forward neural network classifier
- Purpose: Traditional ML baseline

#### **Modern: E5 Classifier**
- Encoder: `intfloat/e5-small-v2` (384-dim, frozen)
- Classification head: Single linear layer
- Optimization: Optuna tunes learning rate, batch size, epochs
- Configuration: n_trials=10, n_splits=3 K-fold CV

#### **State-of-the-art: BERT Full Fine-tuning**
- Model: `bert-base-uncased` (110M parameters)
- Fine-tuning: All layers trainable
- Optimization: Optuna tunes learning rate, batch size, epochs
- Configuration: n_trials=10, n_splits=3 K-fold CV

### **3.2 Training Configuration**
```python
N_TRIALS = 10          # Optuna optimization trials
N_SPLITS = 3           # K-fold cross-validation
SEED = 42              # Reproducibility
MAX_LENGTH = 256       # Token limit
EVAL_RATIO = 0.2       # 20% evaluation split
```

### **3.3 Output**
Models saved to `models/`:
- `tfidf_model/`
- `e5_classifier/`
- `bert_finetuned/`

---

## **4. Method B - LoRA with Domain Adaptation**

**File**: `lora/train_method_b_e5.py`

### **4.1 Architecture Overview**
Three-stage pipeline for parameter-efficient domain adaptation:

**Stage 1: Domain Adapter Training**
- Creates two separate LoRA adapters:
  - IMDB adapter (24k samples)
  - Yelp adapter (24k samples)
- LoRA configuration: Low-rank matrices (r=8 typical)
- Optuna optimization: n_trials=10, n_splits=3 K-fold CV

**Stage 2: Fusion Layer Training**
- Combines IMDB + Yelp adapters
- Learnable gating mechanism
- Trained on combined 48k samples
- Optuna optimization: n_trials=10

**Stage 3: Test-Time Adaptation (TENT)**
- Entropy minimization on Amazon test set
- Adapts frozen model to target domain
- Uses only unlabeled test data

### **4.2 Key Functions**

```python
def tune_and_train_domain_adapter(config, domain_name):
    """
    Creates domain-specific LoRA adapter
    - Loads 24k training samples
    - Optuna tunes: learning_rate, batch_size, epochs
    - Saves best adapter checkpoint
    """
```

```python
def tune_and_train_fusion(config):
    """
    Combines domain adapters with learnable gates
    - Loads IMDB + Yelp adapters
    - Trains fusion layer on 48k combined samples
    - Evaluates on 12k eval set
    """
```

```python
def tent_entropy_minimization(model, test_loader):
    """
    Test-time adaptation via entropy minimization
    - Runs on Amazon test set (6k samples)
    - Updates model parameters to minimize prediction entropy
    - No labels required
    """
```

### **4.3 Training Configuration**
```python
@dataclass
class TrainConfig:
    seed: int = 42
    max_length: int = 256
    eval_ratio: float = 0.2
    n_trials: int = 10
    n_splits: int = 3
```

### **4.4 Output**
Models saved to `outputs/`:
- `domain_adapters/imdb/`
- `domain_adapters/yelp/`
- `fusion_model/`
- `tent_adapted_model/`

---

## **5. Evaluation & Comparison**

**File**: `fine_tuning/third.ipynb`

### **5.1 Metrics Computed**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

### **5.2 Evaluation Scenarios**

**In-Domain Performance**:
- IMDB eval set (6k samples)
- Yelp eval set (6k samples)

**Cross-Domain Transfer**:
- Amazon test set (6k samples)
- Tests generalization to unseen domain

### **5.3 Comparison Framework**
Compares Method A (3 models) vs Method B (LoRA):
- **Parameter efficiency**: LoRA trains <1% of parameters
- **Performance**: Accuracy on in-domain vs cross-domain
- **Adaptation**: TENT test-time adaptation impact

---

## **6. Technical Implementation Details**

### **6.1 Reproducibility**
- **Fixed seed**: 42 everywhere (data sampling, train/eval splits, model initialization)
- **Deterministic operations**: Where possible
- **Identical preprocessing**: Both methods use same data files

### **6.2 Hyperparameter Optimization**
Optuna configuration:
- **n_trials=10**: Number of hyperparameter combinations tested
- **n_splits=3**: K-fold cross-validation splits
- **Objective**: Minimize validation loss
- **Search space**: learning_rate, batch_size, epochs

### **6.3 Model Architectures**

**E5 Embeddings**:
- Base model: `intfloat/e5-small-v2`
- Output dimension: 384
- Frozen encoder (for both methods)

**BERT**:
- Base model: `bert-base-uncased`
- Parameters: ~110M
- Method A: Full fine-tuning
- Method B: LoRA adapters (low-rank)

### **6.4 Data Flow**
```
HuggingFace Datasets
        ↓
first.ipynb (sampling + splitting)
        ↓
../data/processed/
        ↓
   ┌────┴────┐
   ↓         ↓
second.ipynb  train_method_b_e5.py
(Method A)    (Method B)
   ↓         ↓
models/    outputs/
   ↓         ↓
   └────┬────┘
        ↓
third.ipynb (evaluation)
```

---

## **7. Key Innovations & Contributions**

1. **Fair Comparison Framework**: Identical data, seed, and optimization for both methods
2. **IMDB Special Handling**: Discovered and matched LoRA's unique sampling (25k train + 5k test)
3. **Domain Adaptation**: LoRA with fusion layer + TENT test-time adaptation
4. **Comprehensive Evaluation**: In-domain vs cross-domain transfer analysis
5. **Parameter Efficiency**: Compares full fine-tuning vs LoRA (<1% parameters)

---

## **8. File Structure**

```
export_package/
├── data/
│   └── processed/          # Shared datasets (both methods)
│       ├── train.json (48k)
│       ├── eval.json (12k)
│       ├── amazon_test.json (6k)
│       ├── imdb_train.json, imdb_eval.json
│       ├── yelp_train.json, yelp_eval.json
│       └── amazon_train.json, amazon_eval.json
│
├── fine_tuning/            # Method A
│   ├── first.ipynb         # Data preprocessing
│   ├── second.ipynb        # Training (TF-IDF, E5, BERT)
│   ├── third.ipynb         # Evaluation
│   └── models/             # Saved models
│
└── lora/                   # Method B
    ├── train_method_b_e5.py  # LoRA training pipeline
    └── outputs/              # Saved adapters & fusion
```

---

This summary provides a complete overview of the codebase architecture, methodology, and implementation details for your research report.
