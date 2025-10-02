import os, math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from schemas import DistillConfig
from quantize import maybe_quantize_and_export

def _enable_lora_if_set(model, cfg: DistillConfig):
    if cfg.lora_r is None:
        return model
    lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, lora_cfg)

def run_distillation(cfg: DistillConfig, out_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = AutoModelForCausalLM.from_pretrained(cfg.teacher_model, output_hidden_states=True, output_attentions=True).to(device).eval()
    student = AutoModelForCausalLM.from_pretrained(cfg.student_model, output_hidden_states=True, output_attentions=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model if cfg.text_column is None else cfg.student_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA on student (optional)
    student = _enable_lora_if_set(student, cfg)

    ds = load_dataset(cfg.dataset, cfg.subset) if cfg.subset else load_dataset(cfg.dataset)
    split = ds[cfg.split]

    text_col = cfg.text_column or next((k for k, v in split.features.items() if str(v.dtype) == "string"), None)
    if text_col is None:
        text_col = list(split.features.keys())[0]

    def tok_fn(ex):
        return tokenizer(ex[text_col], truncation=True, max_length=512)

    tokenized = split.map(tok_fn, batched=True, remove_columns=split.column_names)
    dc = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dl = torch.utils.data.DataLoader(tokenized, batch_size=cfg.per_device_batch_size, shuffle=True, collate_fn=dc)

    # Optimizer & sched
    optim = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = min(cfg.max_steps, len(dl)) if cfg.max_steps else len(dl)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps)

    kl = nn.KLDivLoss(reduction="batchmean")
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    t = cfg.temperature
    step = 0
    student.train()
    for batch in dl:
        if step >= total_steps: break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            t_out = teacher(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
            t_logits = t_out.logits
            t_h = t_out.hidden_states[-1]
            t_attn = t_out.attentions[-1] if t_out.attentions else None

        s_out = student(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
        s_logits = s_out.logits
        s_h = s_out.hidden_states[-1]
        s_attn = s_out.attentions[-1] if s_out.attentions else None

        # Shift for causal LM loss alignment (predict next token)
        shift_logits = s_logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # KD losses
        s_logp = (s_logits / t).log_softmax(dim=-1)
        t_prob = (t_logits / t).softmax(dim=-1)
        loss_kl = kl(s_logp, t_prob) * (t * t)

        loss_ce = ce(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        loss_mse = mse(s_h, t_h.detach())
        # cosine similarity loss
        cos = nn.functional.cosine_similarity(s_h, t_h.detach(), dim=-1).mean()
        loss_cos = 1.0 - cos

        loss_attn = 0.0
        if s_attn is not None and t_attn is not None:
            loss_attn = mse(s_attn[-1], t_attn[-1].detach())

        loss = (cfg.alpha_kl * loss_kl
                + cfg.beta_ce * loss_ce
                + cfg.gamma_mse * loss_mse
                + cfg.delta_cos * loss_cos
                + cfg.epsilon_attn * loss_attn)

        loss.backward()
        if (step + 1) % cfg.grad_accum_steps == 0:
            optim.step(); sched.step(); optim.zero_grad()

        if (step + 1) % cfg.logging_steps == 0:
            print(f"step={step+1}/{total_steps} loss={loss.item():.4f} kl={loss_kl.item():.4f} ce={loss_ce.item():.4f}")
        if (step + 1) % cfg.save_every == 0:
            student.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        step += 1

    # Final save
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Optional quantization + ONNX
    maybe_quantize_and_export(out_dir, cfg, student, tokenizer, device)
