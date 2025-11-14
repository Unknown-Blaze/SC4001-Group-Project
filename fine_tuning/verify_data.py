import json
from pathlib import Path

print("🔍 Verifying exported data...\n")

required_files = [
    'data/processed/imdb_train.json',
    'data/processed/imdb_val.json',
    'data/processed/imdb_test.json',
    'data/processed/yelp_train.json',
    'data/processed/yelp_val.json',
    'data/processed/yelp_test.json',
    'data/processed/combined_train.json',
    'data/processed/combined_val.json',
    'data/processed/combined_test.json',
    'data/processed/dataset_summary.json'
]

all_good = True
total_samples = 0

for filepath in required_files:
    path = Path(filepath)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            count = len(data)
            total_samples += count
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {filepath}")
            print(f"  {count:,} samples, {size_mb:.1f} MB")
        else:
            print(f"✓ {filepath} (summary file)")
    else:
        print(f"❌ MISSING: {filepath}")
        all_good = False

print(f"\n{'='*60}")
if all_good:
    print(f"✅ All files verified!")
    print(f"📊 Total samples: {total_samples:,}")
else:
    print("⚠️  Some files are missing!")
print(f"{'='*60}")
