# inference.py

import argparse
import torch
import torch.nn.functional as F
from peft import PeftModel

from models_e5 import (
    E5SentimentClassifier,
    E5Backbone,
    FusionSentimentModel,
    get_e5_tokenizer,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_domain_expert(domain_dir: str) -> PeftModel:
    base = E5SentimentClassifier(num_labels=2, train_backbone=False, dropout=0.1)
    model = PeftModel.from_pretrained(base, domain_dir)
    model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_fusion_model(imdb_dir: str,
                       yelp_dir: str,
                       fusion_weights_path: str):
    imdb_expert = load_domain_expert(imdb_dir)
    yelp_expert = load_domain_expert(yelp_dir)
    experts = [imdb_expert, yelp_expert]

    gate_backbone = E5Backbone(trainable=False).to(DEVICE)

    # Hidden dim & dropout are loaded from weights; create with generic values then load.
    fusion_model = FusionSentimentModel(
        experts=experts,
        gate_backbone=gate_backbone,
        num_labels=2,
        hidden_dim=128,
        gate_dropout=0.1,
    ).to(DEVICE)

    state = torch.load(fusion_weights_path, map_location=DEVICE)
    fusion_model.load_state_dict(state, strict=False)
    fusion_model.eval()
    return fusion_model


def predict_sentence(
    text: str,
    imdb_dir: str = "outputs/imdb_adapter",
    yelp_dir: str = "outputs/yelp_adapter",
    fusion_weights_path: str = "outputs/fusion_model/fusion_model_tta.pt",
):
    tokenizer = get_e5_tokenizer()
    fusion_model = build_fusion_model(imdb_dir, yelp_dir, fusion_weights_path)

    enc = tokenizer(
        "query: " + text.strip(),
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        out = fusion_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(probs.argmax())
    label = "positive" if pred_idx == 1 else "negative"
    return label, probs.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True,
                        help="Input sentence to classify")
    parser.add_argument("--imdb_dir", type=str, default="outputs/imdb_adapter")
    parser.add_argument("--yelp_dir", type=str, default="outputs/yelp_adapter")
    parser.add_argument("--fusion_weights", type=str,
                        default="outputs/fusion_model/fusion_model_tta.pt")
    args = parser.parse_args()

    label, probs = predict_sentence(
        args.text,
        imdb_dir=args.imdb_dir,
        yelp_dir=args.yelp_dir,
        fusion_weights_path=args.fusion_weights,
    )
    print(f"Input: {args.text}")
    print(f"Predicted sentiment: {label}")
    print(f"Probabilities [neg, pos]: {probs}")


if __name__ == "__main__":
    main()
