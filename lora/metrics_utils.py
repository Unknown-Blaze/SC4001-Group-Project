# metrics_utils.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict


def compute_classification_metrics(y_true, y_prob, num_labels=2) -> Dict[str, float]:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = y_prob.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
    }

    if num_labels == 2:
        try:
            auc = roc_auc_score(y_true, y_prob[:, 1])
            metrics["auroc"] = float(auc)
        except ValueError:
            metrics["auroc"] = float("nan")

    metrics["ece"] = float(expected_calibration_error(y_true, y_prob, n_bins=15))
    return metrics


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        acc_bin = (y_true[mask] == y_pred[mask]).mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc_bin - conf_bin)
    return ece


@torch.no_grad()
def predict_probs(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
    return all_probs, all_labels
