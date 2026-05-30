"""Model loading + assembly (base LLM, DoRA, norm-unfreeze, selector, patch)."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from leefrag_v2.config import SelectorConfig, V2ModelConfig, V2TrainingConfig
from leefrag_v2.model.patch import install_gated_attention
from leefrag_v2.model.peft_setup import apply_dora, unfreeze_norms
from leefrag_v2.model.selector import LayerSelector

logger = logging.getLogger(__name__)


def load_base(model_config: V2ModelConfig, device):
    """Load the frozen base LLM + tokenizer. HF attn set to 'eager' (its causal
    mask is ignored by the patched attention; eager keeps mask-building cheap)."""
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, model_config.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name, torch_dtype=dtype)
    model.config._attn_implementation = "eager"
    model.to(device)
    return model, tokenizer


def build_teacher_model(model_config: V2ModelConfig, device):
    """Original (un-adapted) model with patched attention, for offline teacher logits."""
    model, tokenizer = load_base(model_config, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    install_gated_attention(model)
    return model, tokenizer


def build_training_model(
    model_config: V2ModelConfig,
    selector_config: SelectorConfig,
    training_config: V2TrainingConfig,
    device,
):
    """Base + DoRA + trainable norms + (learned) selector + patched attention."""
    model, tokenizer = load_base(model_config, device)
    model = apply_dora(model, training_config)
    if training_config.train_norms:
        unfreeze_norms(model)

    selector = None
    if training_config.mode == "learned":
        selector = LayerSelector(
            selector_config, model_config.num_layers, model_config.hidden_size
        ).to(device)

    install_gated_attention(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable (DoRA+norms) params: {n_train/1e6:.1f}M")
    return model, selector, tokenizer
