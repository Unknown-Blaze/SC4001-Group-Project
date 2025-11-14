import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup

import optuna
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm

# Handle TaskType depending on peft version
try:
    from peft import TaskType
except ImportError:
    class TaskType:
        SEQ_CLS = "SEQ_CLS"

from models_e5 import (
    get_e5_tokenizer,
    E5SentimentClassifier,
    E5Backbone,
    FusionSentimentModel,
)
from metrics_utils import (
    compute_classification_metrics,
    predict_probs,
    save_confusion_matrix_plot,
    save_roc_curve,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)
PLOTS_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================
# General utilities
# =========================

@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    num_epochs: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    dropout: float
    weight_decay: float
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    max_length: int = 256


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = "query: " + str(self.texts[idx]).strip()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def load_raw_datasets():
    print("[Main] Loading Hugging Face datasets (IMDB, Yelp, Amazon)...", flush=True)
    imdb = load_dataset("imdb")
    yelp = load_dataset("yelp_polarity")
    amazon = load_dataset("amazon_polarity")
    print("[Main] Datasets loaded.", flush=True)
    return imdb, yelp, amazon


# =========================
# Subsampling to 30k per dataset
# =========================

def build_imdb_30k(imdb, total=30000, eval_ratio=0.2, seed=42):
    """
    IMDB: use all 25k official train + 5k from test (2.5k pos, 2.5k neg).
    Then shuffle and split into train/eval within that 30k.
    """
    rng = np.random.RandomState(seed)

    train = imdb["train"]
    test = imdb["test"]

    train_texts = list(train["text"])
    train_labels = list(train["label"])
    assert len(train_texts) == 25000

    test_labels = np.array(test["label"])
    pos_test_idx = np.where(test_labels == 1)[0]
    neg_test_idx = np.where(test_labels == 0)[0]
    rng.shuffle(pos_test_idx)
    rng.shuffle(neg_test_idx)
    extra_per_class = (total - len(train_texts)) // 2  # 2500

    pos_sel = pos_test_idx[:extra_per_class]
    neg_sel = neg_test_idx[:extra_per_class]

    extra_texts = [test["text"][i] for i in pos_sel] + [test["text"][i] for i in neg_sel]
    extra_labels = [1] * extra_per_class + [0] * extra_per_class

    texts = train_texts + extra_texts
    labels = train_labels + extra_labels

    assert len(texts) == total
    idx = np.arange(total)
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]

    n_eval = int(eval_ratio * total)
    eval_texts = texts[:n_eval]
    eval_labels = labels[:n_eval]
    train_texts = texts[n_eval:]
    train_labels = labels[n_eval:]

    print(
        f"[IMDB] 30k subset: {len(train_texts)} train, {len(eval_texts)} eval "
        f"(pos ratio train={np.mean(train_labels):.3f}, eval={np.mean(eval_labels):.3f})",
        flush=True,
    )

    return train_texts, train_labels, eval_texts, eval_labels


def build_yelp_30k(yelp, total=30000, eval_ratio=0.2, seed=42):
    """
    Yelp: sample 15k pos + 15k neg from train, unique.
    Then shuffle and split into train/eval.
    """
    rng = np.random.RandomState(seed)
    train = yelp["train"]
    labels = np.array(train["label"])
    texts = np.array(train["text"])

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    per_class = total // 2  # 15000
    pos_sel = pos_idx[:per_class]
    neg_sel = neg_idx[:per_class]

    sel_idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel_idx)

    texts_sel = texts[sel_idx].tolist()
    labels_sel = labels[sel_idx].tolist()

    n_eval = int(eval_ratio * total)
    eval_texts = texts_sel[:n_eval]
    eval_labels = labels_sel[:n_eval]
    train_texts = texts_sel[n_eval:]
    train_labels = labels_sel[n_eval:]

    print(
        f"[YELP] 30k subset: {len(train_texts)} train, {len(eval_texts)} eval "
        f"(pos ratio train={np.mean(train_labels):.3f}, eval={np.mean(eval_labels):.3f})",
        flush=True,
    )

    return train_texts, train_labels, eval_texts, eval_labels


def build_amazon_30k(amazon, total=30000, adapt_ratio=2.0 / 3.0, seed=42):
    """
    Amazon: sample 15k pos + 15k neg from train, unique.
    Then shuffle and split into:
      - adapt set for TTA (~20k)
      - eval set (~10k)
    """
    rng = np.random.RandomState(seed)
    train = amazon["train"]
    labels = np.array(train["label"])
    text_field = "text" if "text" in train.column_names else "content"
    texts = np.array(train[text_field])

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    per_class = total // 2  # 15000
    pos_sel = pos_idx[:per_class]
    neg_sel = neg_idx[:per_class]

    sel_idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(sel_idx)

    texts_sel = texts[sel_idx].tolist()
    labels_sel = labels[sel_idx].tolist()

    n_adapt = int(adapt_ratio * total)  # 20000
    adapt_texts = texts_sel[:n_adapt]
    adapt_labels = labels_sel[:n_adapt]
    eval_texts = texts_sel[n_adapt:]
    eval_labels = labels_sel[n_adapt:]

    print(
        f"[AMAZON] 30k subset: {len(adapt_texts)} adapt, {len(eval_texts)} eval "
        f"(pos ratio adapt={np.mean(adapt_labels):.3f}, eval={np.mean(eval_labels):.3f})",
        flush=True,
    )

    return adapt_texts, adapt_labels, eval_texts, eval_labels


# =========================
# Model builders & training helpers
# =========================

def make_lora_e5_classifier(cfg: TrainConfig) -> PeftModel:
    base = E5SentimentClassifier(
        num_labels=2,
        train_backbone=False,
        dropout=cfg.dropout,
    )
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(base, lora_config)
    return model


def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    scheduler,
                    cfg: TrainConfig,
                    desc: str):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate_on_texts(model,
                      texts,
                      labels,
                      tokenizer,
                      max_length: int = 256,
                      batch_size: int = 64,
                      tag: str = "",
                      plot_prefix: str = None) -> Dict:
    """
    Run evaluation on (texts, labels), return metrics, and optionally save plots.
    """
    ds = TextDataset(texts, labels, tokenizer, max_length=max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs, y_true = predict_probs(model, loader, DEVICE)
    metrics = compute_classification_metrics(y_true, probs, num_labels=2)
    print(f"[Eval][{tag}] {metrics}", flush=True)

    if plot_prefix is not None:
        y_pred = np.argmax(probs, axis=1)
        cm_path = f"{plot_prefix}_cm.png"
        roc_path = f"{plot_prefix}_roc.png"
        save_confusion_matrix_plot(
            y_true,
            y_pred,
            labels=[0, 1],
            out_path=cm_path,
            title=f"{tag} - Confusion Matrix",
        )
        save_roc_curve(
            y_true,
            probs[:, 1],
            out_path=roc_path,
            title=f"{tag} - ROC Curve",
        )

    return metrics


# =========================
# Adapter tuning with Optuna + K-fold
# =========================

def tune_and_train_domain_adapter(
    domain_name: str,
    texts: List[str],
    labels: List[int],
    n_splits: int = 3,
    n_trials: int = 10,
    output_dir: str = "outputs",
) -> Tuple[str, TrainConfig]:
    set_seed(42)
    tokenizer = get_e5_tokenizer()
    print(f"[{domain_name.upper()}] Tuning adapter on {len(texts)} samples...", flush=True)

    texts = np.array(texts)
    labels = np.array(labels)

    def objective(trial):
        cfg = TrainConfig(
            lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
            batch_size=trial.suggest_categorical("batch_size", [64]),
            num_epochs=trial.suggest_int("num_epochs", 2, 15),
            lora_r=trial.suggest_categorical("lora_r", [8, 16, 32]),
            lora_alpha=trial.suggest_categorical("lora_alpha", [16, 32, 64]),
            lora_dropout=trial.suggest_float("lora_dropout", 0.05, 0.3),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )

        print(f"[{domain_name.upper()}][Trial {trial.number}] cfg={cfg}", flush=True)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            print(
                f"[{domain_name.upper()}][Trial {trial.number}] "
                f"Fold {fold_idx+1}/{n_splits}...",
                flush=True,
            )

            train_ds = TextDataset(
                texts[train_idx].tolist(),
                labels[train_idx].tolist(),
                tokenizer,
                max_length=cfg.max_length,
            )
            val_ds = TextDataset(
                texts[val_idx].tolist(),
                labels[val_idx].tolist(),
                tokenizer,
                max_length=cfg.max_length,
            )

            train_loader = DataLoader(
                train_ds,
                batch_size=cfg.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
            )

            model = make_lora_e5_classifier(cfg).to(DEVICE)

            num_steps = cfg.num_epochs * len(train_loader)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(cfg.warmup_ratio * num_steps),
                num_training_steps=num_steps,
            )

            for epoch in range(cfg.num_epochs):
                desc = (
                    f"[{domain_name}][T{trial.number}]"
                    f"[F{fold_idx+1}] Ep {epoch+1}/{cfg.num_epochs}"
                )
                loss = train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    cfg,
                    desc,
                )
                print(
                    f"[{domain_name.upper()}][Trial {trial.number}] "
                    f"Fold {fold_idx+1} Ep {epoch+1} Loss={loss:.4f}",
                    flush=True,
                )

            # Evaluate on val fold
            val_probs, val_labels = predict_probs(model, val_loader, DEVICE)
            fold_metrics = compute_classification_metrics(
                val_labels,
                val_probs,
                num_labels=2,
            )
            print(
                f"[{domain_name.upper()}][Trial {trial.number}] "
                f"Fold {fold_idx+1} Val macro-F1={fold_metrics['macro_f1']:.4f}",
                flush=True,
            )
            f1_scores.append(fold_metrics["macro_f1"])

            del model
            torch.cuda.empty_cache()

        mean_f1 = float(np.mean(f1_scores))
        print(
            f"[{domain_name.upper()}][Trial {trial.number}] "
            f"Mean CV macro-F1={mean_f1:.4f}",
            flush=True,
        )
        return mean_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial.params
    best_cfg = TrainConfig(
        lr=best["lr"],
        batch_size=best["batch_size"],
        num_epochs=best["num_epochs"],
        lora_r=best["lora_r"],
        lora_alpha=best["lora_alpha"],
        lora_dropout=best["lora_dropout"],
        dropout=best["dropout"],
        weight_decay=best["weight_decay"],
    )

    print(f"[{domain_name.upper()}] Best params: {best}", flush=True)

    # Final training
    full_ds = TextDataset(
        texts.tolist(),
        labels.tolist(),
        tokenizer,
        max_length=best_cfg.max_length,
    )
    full_loader = DataLoader(
        full_ds,
        batch_size=best_cfg.batch_size,
        shuffle=True,
    )

    model = make_lora_e5_classifier(best_cfg).to(DEVICE)
    num_steps = best_cfg.num_epochs * len(full_loader)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_cfg.lr,
        weight_decay=best_cfg.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(best_cfg.warmup_ratio * num_steps),
        num_training_steps=num_steps,
    )

    print(f"[{domain_name.upper()}] Training final adapter...", flush=True)
    for epoch in range(best_cfg.num_epochs):
        desc = f"[{domain_name}][FINAL] Ep {epoch+1}/{best_cfg.num_epochs}"
        loss = train_one_epoch(
            model,
            full_loader,
            optimizer,
            scheduler,
            best_cfg,
            desc,
        )
        print(
            f"[{domain_name.upper()}][FINAL] Ep {epoch+1} Loss={loss:.4f}",
            flush=True,
        )

    domain_dir = os.path.join(output_dir, f"{domain_name}_adapter")
    os.makedirs(domain_dir, exist_ok=True)
    model.save_pretrained(domain_dir)

    with open(os.path.join(domain_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    print(f"[{domain_name.upper()}] Saved adapter to {domain_dir}", flush=True)
    return domain_dir, best_cfg


def load_domain_expert(domain_dir: str) -> PeftModel:
    base = E5SentimentClassifier(
        num_labels=2,
        train_backbone=False,
        dropout=0.1,
    )
    model = PeftModel.from_pretrained(base, domain_dir)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# =========================
# Fusion tuning and training
# =========================

def tune_and_train_fusion(
    imdb_dir: str,
    yelp_dir: str,
    imdb_train_texts,
    imdb_train_labels,
    yelp_train_texts,
    yelp_train_labels,
    n_trials: int = 15,
    output_dir: str = "outputs",
):
    print("[Fusion] Building pooled train set from IMDB+Yelp 30k subsets...", flush=True)
    tokenizer = get_e5_tokenizer()

    all_texts = list(imdb_train_texts) + list(yelp_train_texts)
    all_labels = list(imdb_train_labels) + list(yelp_train_labels)

    rng = np.random.RandomState(42)
    indices = np.arange(len(all_texts))
    rng.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TextDataset(
        [all_texts[i] for i in train_idx],
        [all_labels[i] for i in train_idx],
        tokenizer,
        max_length=256,
    )
    val_ds = TextDataset(
        [all_texts[i] for i in val_idx],
        [all_labels[i] for i in val_idx],
        tokenizer,
        max_length=256,
    )

    imdb_expert = load_domain_expert(imdb_dir)
    yelp_expert = load_domain_expert(yelp_dir)
    experts = [imdb_expert, yelp_expert]
    gate_backbone = E5Backbone(trainable=False).to(DEVICE)

    print("[Fusion] Starting Optuna tuning...", flush=True)

    def build_fusion(hidden_dim, gate_dropout):
        return FusionSentimentModel(
            experts=experts,
            gate_backbone=gate_backbone,
            num_labels=2,
            hidden_dim=hidden_dim,
            gate_dropout=gate_dropout,
        ).to(DEVICE)

    def objective(trial):
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        gate_dropout = trial.suggest_float("gate_dropout", 0.0, 0.3)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        num_epochs = trial.suggest_int("num_epochs", 2, 15)
        batch_size = trial.suggest_categorical("batch_size", [64])

        print(
            f"[Fusion][Trial {trial.number}] "
            f"hd={hidden_dim}, drop={gate_dropout:.3f}, "
            f"lr={lr:.2e}, wd={weight_decay:.2e}, "
            f"ep={num_epochs}, bs={batch_size}",
            flush=True,
        )

        fusion_model = build_fusion(hidden_dim, gate_dropout)
        optimizer = torch.optim.AdamW(
            fusion_model.gate.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            fusion_model.train()
            epoch_loss = 0.0
            desc = f"[Fusion][T{trial.number}] Ep {epoch+1}/{num_epochs}"
            for batch in tqdm(train_loader, desc=desc, leave=False):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                out = fusion_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.gate.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * labels.size(0)
            epoch_loss /= len(train_ds)
            print(
                f"[Fusion][Trial {trial.number}] Ep {epoch+1} Loss={epoch_loss:.4f}",
                flush=True,
            )

        val_probs, val_labels = predict_probs(fusion_model, val_loader, DEVICE)
        metrics = compute_classification_metrics(val_labels, val_probs, num_labels=2)
        print(
            f"[Fusion][Trial {trial.number}] Val macro-F1={metrics['macro_f1']:.4f}",
            flush=True,
        )

        del fusion_model
        torch.cuda.empty_cache()

        return metrics["macro_f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial.params
    print("[Fusion] Best params:", best, flush=True)

    # Final training
    hidden_dim = best["hidden_dim"]
    gate_dropout = best["gate_dropout"]
    lr = best["lr"]
    weight_decay = best["weight_decay"]
    num_epochs = best["num_epochs"]
    batch_size = best["batch_size"]

    full_ds = TextDataset(all_texts, all_labels, tokenizer, max_length=256)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)

    imdb_expert = load_domain_expert(imdb_dir)
    yelp_expert = load_domain_expert(yelp_dir)
    experts = [imdb_expert, yelp_expert]
    gate_backbone = E5Backbone(trainable=False).to(DEVICE)

    fusion_model = FusionSentimentModel(
        experts=experts,
        gate_backbone=gate_backbone,
        num_labels=2,
        hidden_dim=hidden_dim,
        gate_dropout=gate_dropout,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        fusion_model.gate.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    print("[Fusion] Training final fusion model...", flush=True)
    for epoch in range(num_epochs):
        fusion_model.train()
        total_loss = 0.0
        desc = f"[Fusion][FINAL] Ep {epoch+1}/{num_epochs}"
        for batch in tqdm(full_loader, desc=desc, leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            out = fusion_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.gate.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        total_loss /= len(full_ds)
        print(
            f"[Fusion][FINAL] Ep {epoch+1} Loss={total_loss:.4f}",
            flush=True,
        )

    fusion_dir = os.path.join(output_dir, "fusion_model")
    os.makedirs(fusion_dir, exist_ok=True)
    torch.save(
        fusion_model.state_dict(),
        os.path.join(fusion_dir, "fusion_model.pt"),
    )
    with open(os.path.join(fusion_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    print(f"[Fusion] Saved fusion model to {fusion_dir}", flush=True)
    return fusion_model, fusion_dir


# =========================
# Multi-step TENT-style adaptation
# =========================

def tent_entropy_minimization_steps(
    fusion_model,
    adapt_texts,
    eval_texts,
    eval_labels,
    tokenizer,
    steps: int = 10,
    batch_size: int = 64,
    lr: float = 1e-4,
    max_adapt_samples: int = 5000,
    output_dir: str = "outputs/fusion_model_tta_steps",
):
    """
    Multi-step TENT-style adaptation (entropy minimization):

    - Only gate parameters are updated (experts + backbone frozen).
    - Each step: one pass over (subset of) adapt_texts.
    - After each step: evaluate on eval_texts/labels.
    - For each step:
        - Log metrics
        - Save confusion matrix + ROC curve
        - Save model weights

    Returns:
        metrics_per_step: dict[int -> metrics]
        (step 0 is pre-TTA baseline)
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[TTA] Starting multi-step TENT adaptation for {steps} steps...", flush=True)

    # Adaptation subset for speed/stability
    use_adapt = adapt_texts[: min(len(adapt_texts), max_adapt_samples)]

    class AmazonAdaptDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=256):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = "query: " + str(self.texts[idx]).strip()
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }

    adapt_ds = AmazonAdaptDataset(use_adapt, tokenizer)
    adapt_loader = DataLoader(adapt_ds, batch_size=batch_size, shuffle=True)

    # Freeze everything except gate
    for p in fusion_model.parameters():
        p.requires_grad = False
    for p in fusion_model.gate.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(fusion_model.gate.parameters(), lr=lr)

    def entropy_mean(probs):
        return - (probs * (probs + 1e-12).log()).sum(dim=-1).mean()

    def eval_on_amazon(step: int | None):
        ds_eval = TextDataset(eval_texts, eval_labels, tokenizer, max_length=256)
        eval_loader = DataLoader(ds_eval, batch_size=batch_size, shuffle=False)
        probs, y_true = predict_probs(fusion_model, eval_loader, DEVICE)
        metrics = compute_classification_metrics(y_true, probs, num_labels=2)

        tag = "pre-TTA" if step is None else f"post-TTA step {step}"
        print(f"[TTA][{tag}] {metrics}", flush=True)

        y_pred = np.argmax(probs, axis=1)
        if step is None:
            cm_path = os.path.join(output_dir, "amazon_tta_pre_cm.png")
            roc_path = os.path.join(output_dir, "amazon_tta_pre_roc.png")
        else:
            cm_path = os.path.join(output_dir, f"amazon_tta_step_{step}_cm.png")
            roc_path = os.path.join(output_dir, f"amazon_tta_step_{step}_roc.png")

        save_confusion_matrix_plot(
            y_true,
            y_pred,
            labels=[0, 1],
            out_path=cm_path,
            title=f"Amazon {tag} - Confusion Matrix",
        )
        save_roc_curve(
            y_true,
            probs[:, 1],
            out_path=roc_path,
            title=f"Amazon {tag} - ROC Curve",
        )

        return metrics

    metrics_per_step: Dict[int, Dict] = {}

    # Step 0: baseline
    metrics_per_step[0] = eval_on_amazon(step=None)

    fusion_model.train()

    # Steps 1..N: cumulative adaptation
    for step in range(1, steps + 1):
        step_loss = 0.0
        for batch in tqdm(adapt_loader, desc=f"[TTA] Step {step}/{steps}", leave=False):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            with torch.no_grad():
                pooled = fusion_model.gate_backbone(input_ids, attention_mask)
                expert_logits = []
                for expert in fusion_model.experts:
                    out = expert(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out["logits"]
                    expert_logits.append(logits.unsqueeze(1))
                expert_logits = torch.cat(expert_logits, dim=1)

            gate_weights = fusion_model.gate(pooled).unsqueeze(-1)
            fused_logits = (expert_logits * gate_weights).sum(1)
            probs = torch.softmax(fused_logits, dim=-1)

            loss = entropy_mean(probs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.gate.parameters(), 1.0)
            optimizer.step()

            step_loss += loss.item() * input_ids.size(0)

        step_loss /= len(adapt_ds)
        print(f"[TTA] Step {step} entropy loss={step_loss:.6f}", flush=True)

        # Evaluate and store metrics for this step
        metrics_step = eval_on_amazon(step=step)
        metrics_per_step[step] = metrics_step

        # Save model weights at this step
        torch.save(
            fusion_model.state_dict(),
            os.path.join(output_dir, f"fusion_model_tta_step_{step}.pt"),
        )

    print("[TTA] Finished multi-step TENT adaptation.", flush=True)
    return metrics_per_step


# =========================
# Main
# =========================

def main():
    set_seed(42)
    imdb_raw, yelp_raw, amazon_raw = load_raw_datasets()

    # Build controlled 30k subsets
    imdb_train_texts, imdb_train_labels, imdb_eval_texts, imdb_eval_labels = build_imdb_30k(imdb_raw)
    yelp_train_texts, yelp_train_labels, yelp_eval_texts, yelp_eval_labels = build_yelp_30k(yelp_raw)
    amazon_adapt_texts, amazon_adapt_labels, amazon_eval_texts, amazon_eval_labels = build_amazon_30k(amazon_raw)

    # 1) Per-domain adapters
    imdb_dir, _ = tune_and_train_domain_adapter(
        "imdb",
        imdb_train_texts,
        imdb_train_labels,
        n_splits=3,
        n_trials=15,
        output_dir="outputs",
    )
    yelp_dir, _ = tune_and_train_domain_adapter(
        "yelp",
        yelp_train_texts,
        yelp_train_labels,
        n_splits=3,
        n_trials=15,
        output_dir="outputs",
    )

    # 2) Fusion model
    fusion_model, fusion_dir = tune_and_train_fusion(
        imdb_dir,
        yelp_dir,
        imdb_train_texts,
        imdb_train_labels,
        yelp_train_texts,
        yelp_train_labels,
        n_trials=15,
        output_dir="outputs",
    )

    tokenizer = get_e5_tokenizer()
    metrics = {}

    # 3) Eval on source domains
    metrics["fusion_imdb_eval"] = evaluate_on_texts(
        fusion_model,
        imdb_eval_texts,
        imdb_eval_labels,
        tokenizer,
        tag="IMDB eval (subset of 30k)",
        plot_prefix=os.path.join(PLOTS_DIR, "imdb_eval"),
    )
    metrics["fusion_yelp_eval"] = evaluate_on_texts(
        fusion_model,
        yelp_eval_texts,
        yelp_eval_labels,
        tokenizer,
        tag="Yelp eval (subset of 30k)",
        plot_prefix=os.path.join(PLOTS_DIR, "yelp_eval"),
    )

    # 4) Amazon pre-TTA
    metrics["fusion_amazon_before_tta"] = evaluate_on_texts(
        fusion_model,
        amazon_eval_texts,
        amazon_eval_labels,
        tokenizer,
        tag="Amazon eval pre-TTA (subset of 30k)",
        plot_prefix=os.path.join(PLOTS_DIR, "amazon_pre_tta"),
    )

    # 5) Multi-step TTA on Amazon
    tta_steps = 20  # you can change this
    tta_metrics = tent_entropy_minimization_steps(
        fusion_model,
        amazon_adapt_texts,
        amazon_eval_texts,
        amazon_eval_labels,
        tokenizer,
        steps=tta_steps,
        batch_size=64,
        lr=7e-5,
        max_adapt_samples=5000,
        output_dir=os.path.join(fusion_dir, "tta_steps"),
    )
    metrics["fusion_amazon_tta_steps"] = tta_metrics

    # Save final (last-step) adapted fusion model
    torch.save(
        fusion_model.state_dict(),
        os.path.join(fusion_dir, "fusion_model_tta_final.pt"),
    )

    # 6) Save metrics
    with open(os.path.join("outputs", "method_b_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== Method B Metrics (stored in outputs/method_b_metrics.json) ===", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
