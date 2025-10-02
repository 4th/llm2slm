import os, torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from schemas import DistillConfig

def _export_onnx(model, tokenizer, out_dir, device):
    try:
        import onnx
        from transformers.onnx import export, FeaturesManager
        onnx_path = os.path.join(out_dir, "model.onnx")
        task = "causal-lm"
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=task)
        onnx_config = model_onnx_config(model.config)
        export(tokenizer, model, onnx_config, opset=17, output=onnx_path)
    except Exception as e:
        print(f"ONNX export skipped: {e}")

def maybe_quantize_and_export(out_dir: str, cfg: DistillConfig, model=None, tokenizer=None, device="cpu"):
    # Simple PTQ (dynamic) for CPU
    if cfg.post_training_quant:
        try:
            model.to("cpu")
            model.eval()
            qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            q_path = os.path.join(out_dir, "int8-dynamic")
            os.makedirs(q_path, exist_ok=True)
            qmodel.save_pretrained(q_path)
            if tokenizer:
                tokenizer.save_pretrained(q_path)
        except Exception as e:
            print(f"Post-training quantization skipped: {e}")

    if cfg.export_onnx:
        try:
            _export_onnx(model, tokenizer, out_dir, device)
        except Exception as e:
            print(f"ONNX export failed: {e}")
