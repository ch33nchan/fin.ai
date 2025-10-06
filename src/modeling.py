from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from peft import LoraConfig, PeftModel, get_peft_model


class MeanPooler(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(hidden_states * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1)
        return summed / denom


class RerankerModel(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int, base_model_name: str):
        super().__init__()
        self.encoder = base_model
        self.pooler = MeanPooler()
        self.score_head = nn.Linear(hidden_size, 1)
    
        self.base_model_name = base_model_name

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        scores = self.score_head(pooled)
        return scores.squeeze(-1)

    def save_pretrained(self, output_dir: str) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.encoder, "save_pretrained"):
            self.encoder.save_pretrained(output_path / "encoder")
        torch.save(self.score_head.state_dict(), output_path / "score_head.pt")
        info = {
            "base_model_name": self.base_model_name,
        }
        with (output_path / "model_info.json").open("w", encoding="utf-8") as f:
            json.dump(info, f)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "RerankerModel":
        model_path = Path(model_dir)
        info_path = model_path / "model_info.json"
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as f:
                info = json.load(f)
            base_model_name = info.get("base_model_name")
        else:
            base_model_name = None

        encoder_dir = model_path / "encoder"
        adapter_config = encoder_dir / "adapter_config.json"
        if adapter_config.exists():
            if base_model_name is None:
                raise ValueError("LoRA adapter found but base_model_name missing in model_info.json")
            base_encoder = AutoModel.from_pretrained(base_model_name)
            encoder = PeftModel.from_pretrained(base_encoder, encoder_dir)
        else:
            encoder = AutoModel.from_pretrained(encoder_dir)
            if base_model_name is None:
                base_model_name = encoder.config._name_or_path

        hidden_size = encoder.config.hidden_size
        model = cls(encoder, hidden_size, base_model_name)
        state_dict = torch.load(model_path / "score_head.pt", map_location="cpu")
        model.score_head.load_state_dict(state_dict)
        return model


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def build_model(
    model_id: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Tuple[str, ...] = ("query", "key", "value"),
    torch_dtype: Optional[torch.dtype] = None,
) -> RerankerModel:
    config = AutoConfig.from_pretrained(model_id)
    base_model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype, config=config)

    if lora_r > 0:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias="none",
            task_type="SEQ_CLS",
        )
        base_model = get_peft_model(base_model, lora_cfg)

    model = RerankerModel(base_model, hidden_size=config.hidden_size, base_model_name=model_id)
    return model


def freeze_encoder_layers(model: nn.Module, trainable_layers: int = 2) -> None:
    if not hasattr(model, "encoder"):
        return
    encoder = model.encoder
    if not hasattr(encoder, "encoder"):
        return
    layers = encoder.encoder.layer
    for layer in layers[:-trainable_layers]:
        for param in layer.parameters():
            param.requires_grad = False
