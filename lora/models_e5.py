import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List

E5_MODEL_NAME = "intfloat/e5-small-v2"


def get_e5_tokenizer():
    return AutoTokenizer.from_pretrained(E5_MODEL_NAME)


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling as recommended for e5-style models.
    """
    mask = attention_mask.unsqueeze(-1).bool()
    masked = last_hidden_states.masked_fill(~mask, 0.0)
    denom = mask.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / denom


class E5Backbone(nn.Module):
    """
    Wrapper around intfloat/e5-small-v2 with optional freezing.
    """
    def __init__(self, trainable: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(E5_MODEL_NAME)
        if not trainable:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled = average_pool(outputs.last_hidden_state, attention_mask)
        return pooled


class E5SentimentClassifier(nn.Module):
    """
    Sentiment classifier on top of e5; PEFT-compatible.

    Key: exposes .config and accepts HF-style forward kwargs so PEFT can wrap it.
    """
    def __init__(self,
                 num_labels: int = 2,
                 train_backbone: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.backbone = E5Backbone(trainable=train_backbone)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.hidden_size, num_labels)

        # Make it look like a Hugging Face model for PEFT / Trainer.
        self.config = self.backbone.encoder.config
        self.config.num_labels = num_labels
        self.config.problem_type = "single_label_classification"

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
        **kwargs
    ):
        """
        Support both (input_ids, attention_mask) and (inputs_embeds, attention_mask),
        because PEFT / HF sometimes route via inputs_embeds.
        """
        if inputs_embeds is not None:
            # Directly use underlying encoder with provided embeddings
            outputs = self.backbone.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )
            pooled = average_pool(outputs.last_hidden_state, attention_mask)
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            pooled = self.backbone(input_ids, attention_mask)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Always return a dict (PEFT expects mapping-style outputs)
        return {"loss": loss, "logits": logits}


class FusionGate(nn.Module):
    """
    Gating MLP that outputs mixture weights over experts.
    """
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, pooled):
        logits = self.net(pooled)
        return torch.softmax(logits, dim=-1)


class FusionSentimentModel(nn.Module):
    """
    Late-logit fusion of frozen expert models with a learned gate.

    - experts: list of modules returning dict with "logits"
    - gate_backbone: frozen E5Backbone to compute pooled representation for gating
    """
    def __init__(self,
                 experts: List[nn.Module],
                 gate_backbone: E5Backbone,
                 num_labels: int = 2,
                 hidden_dim: int = 128,
                 gate_dropout: float = 0.1):
        super().__init__()

        self.experts = nn.ModuleList(experts)
        for e in self.experts:
            for p in e.parameters():
                p.requires_grad = False

        self.gate_backbone = gate_backbone
        for p in self.gate_backbone.parameters():
            p.requires_grad = False

        self.num_experts = len(experts)
        self.num_labels = num_labels

        self.gate = FusionGate(
            input_dim=self.gate_backbone.hidden_size,
            num_experts=self.num_experts,
            hidden_dim=hidden_dim,
            dropout=gate_dropout,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        # Get pooled embedding (no grad through backbone)
        with torch.no_grad():
            pooled = self.gate_backbone(input_ids, attention_mask)

        # Collect expert logits
        expert_logits = []
        for expert in self.experts:
            out = expert(input_ids=input_ids,
                         attention_mask=attention_mask)
            logits = out["logits"] if isinstance(out, dict) else out
            expert_logits.append(logits.unsqueeze(1))  # [B,1,C]

        expert_logits = torch.cat(expert_logits, dim=1)       # [B,K,C]
        gate_weights = self.gate(pooled).unsqueeze(-1)        # [B,K,1]
        fused_logits = (expert_logits * gate_weights).sum(1)  # [B,C]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(fused_logits, labels)

        return {
            "loss": loss,
            "logits": fused_logits,
            "gate_weights": gate_weights.squeeze(-1),
        }
