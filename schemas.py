from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class DistillConfig(BaseModel):
    teacher_model: str = Field(..., description="HF hub id or path for the teacher")
    student_model: str = Field(..., description="HF hub id or path for the student")
    dataset: str = Field(..., description="HF dataset id, e.g., 'wikitext'")
    subset: Optional[str] = None
    split: str = "train"
    text_column: Optional[str] = None

    max_steps: int = 1000
    per_device_batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    grad_accum_steps: int = 8
    logging_steps: int = 20
    save_every: int = 200

    temperature: float = 2.0
    alpha_kl: float = 0.7
    beta_ce: float = 0.3
    gamma_mse: float = 0.05
    delta_cos: float = 0.05
    epsilon_attn: float = 0.0

    lora_r: Optional[int] = 8
    lora_alpha: Optional[int] = 16
    lora_dropout: Optional[float] = 0.05

    use_qat: bool = False
    post_training_quant: bool = True
    export_onnx: bool = True

class JobStatus(BaseModel):
    job_id: str
    state: str
    message: str = ""
    artifacts: Optional[List[str]] = None
