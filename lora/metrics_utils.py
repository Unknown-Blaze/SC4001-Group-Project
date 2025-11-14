import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt


def predict_probs(model, dataloader, device):
    """
    Run model on dataloader and return:
      - probs: [N, C]
      - labels: [N] (if present in batch)
    Assumes model returns dict with 'logits'.
    """
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

            if "labels" in batch:
                labels = batch["labels"].numpy()
                all_labels.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)
    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_labels = None
    return all_probs, all_labels


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE) using max prob binning.
    """
    y_true = np.asarray(y_true)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        start = bin_boundaries[i]
        end = bin_boundaries[i + 1]
        if i == 0:
            mask = (confidences >= start) & (confidences <= end)
        else:
            mask = (confidences > start) & (confidences <= end)

        if not np.any(mask):
            continue

        acc = (y_true[mask] == predictions[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def compute_classification_metrics(y_true, y_prob, num_labels: int = 2):
    """
    Compute a rich set of metrics for classification.

    Returns dict with:
      - accuracy
      - macro_f1, micro_f1
      - macro_precision, micro_precision
      - macro_recall, micro_recall
      - auroc (binary or multiclass OVR)
      - ece
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }

    # AUROC
    try:
        if num_labels == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        metrics["auroc"] = float("nan")

    # ECE
    metrics["ece"] = float(expected_calibration_error(y_true, y_prob))

    return metrics


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    labels,
    out_path: str,
    title: str = "Confusion Matrix",
):
    """
    Save a confusion matrix heatmap as PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(
    y_true,
    y_score,
    out_path: str,
    title: str = "ROC Curve",
):
    """
    Save ROC curve plot as PNG for binary classification.
    y_score is the predicted probability for the positive class.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        # If only one class present or degenerate case
        fpr = np.array([0.0, 1.0])
        tpr = np.array([0.0, 1.0])
        roc_auc = float("nan")

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
