# Server Setup Instructions

## 1. Transfer Files
Upload the entire `export_package` folder to your server.

## 2. Create Virtual Environment
```bash
# On server (Linux/Mac)
python3 -m venv .venv
source .venv/bin/activate

# Or on Windows server
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt

# For GPU support (if CUDA available)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 4. Verify Data
```bash
python verify_data.py
```

## 5. Run Training
```bash
# Open Jupyter
jupyter notebook

# Or run directly
jupyter nbconvert --execute second.ipynb
```

## Directory Structure After Setup
```
your_project/
├── data/
│   └── processed/          # All preprocessed datasets (DO NOT MODIFY)
├── first.ipynb            # Data preparation (already executed)
├── second.ipynb           # Model training (run this)
├── third.ipynb            # Evaluation (run this)
├── requirements.txt
└── .venv/                 # Virtual environment (create on server)
```

## Expected Data Files
- `data/processed/imdb_train.json` (35k reviews)
- `data/processed/imdb_val.json` (7.5k reviews)
- `data/processed/imdb_test.json` (7.5k reviews)
- `data/processed/yelp_train.json` (~26k reviews)
- `data/processed/yelp_val.json` (~5.6k reviews)
- `data/processed/yelp_test.json` (~5.6k reviews)
- `data/processed/combined_train.json` (~61k reviews)
- `data/processed/combined_val.json` (~13k reviews)
- `data/processed/combined_test.json` (10k Amazon reviews)
- `data/processed/dataset_summary.json`

## Disk Space Requirements
- **Processed Data**: ~300-500 MB
- **Model Checkpoints**: ~1-2 GB (during training)
- **Python Packages**: ~2-3 GB
- **Total Required**: ~5-6 GB free space

## Notes
- Raw datasets (IMDB, Yelp, Amazon) are NOT included (already processed)
- Virtual environment (.venv) must be recreated on server
- All data is in `data/processed/` - ready to use
