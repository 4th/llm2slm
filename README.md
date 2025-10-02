# LLM→SLM Distillation Microservice (Kubernetes‑ready)

Shrink a Large Language Model (LLM, the **teacher**) into a Small Language Model (SLM, the **student**) using a hybrid knowledge‑distillation algorithm. This service exposes a REST API, writes artifacts to persistent storage, and ships with Kubernetes manifests for quick deployment.

> **Status:** Developer preview. Verified on Windows (Python 3.13) and Linux. GPU optional.

---

## Architecture

```mermaid
flowchart LR
  %% Client
  U[Client<br/>cURL / CLI / UI] -->|POST /distill| API

  %% Cluster
  subgraph K8s["Kubernetes Cluster"]
    API[FastAPI API<br/>Uvicorn] -->|enqueue/run| JOB[Distillation Job/Worker]
    API --> PVC[(PVC: /data)]
    JOB --> PVC
    CM[(ConfigMap)] -. defaults .-> API
  end

  %% External deps
  JOB -->|Teacher/Student Models| HF[(Hugging Face Hub)]
  JOB -->|Datasets| DS[(Datasets Hub)]

  %% Artifacts
  subgraph Artifacts
    ART[(Student Weights<br/>Tokenizer<br/>int8-dynamic/<br/>model.onnx)]
  end
  PVC --> ART
  API -->|GET /status, /artifacts, /download| ART
```

**Flow:**
1. Client calls `POST /distill` with teacher/student/dataset config.
2. API kicks off a distillation run (Job/Worker).  
3. Worker pulls models/datasets, trains student with KD losses.  
4. Artifacts (student weights, tokenizer, optional int8 + ONNX) are persisted to a PVC.  
5. Client fetches status & downloads artifacts via the API.

---

## Why LLM→SLM?

- **Latency & cost:** smaller models are faster and cheaper to serve.
- **Edge & offline:** deploy on constrained hardware.
- **Domain refresh:** frequently update a compact student from an evolving teacher.

---

## Key Features

- **Hybrid KD loss** — temperature‑scaled KL (soft targets) + CE (hard pseudo‑labels) + hidden‑state MSE + cosine alignment; optional attention‑map MSE. [1–4, 14]  
- **Parameter‑efficient finetuning** — optional **LoRA** on the student. [5]  
- **Quantization & export** — post‑training dynamic INT8 and **ONNX** export (best‑effort). [6–8]  
- **Kubernetes‑ready** — PVC, ConfigMap, Deployment/Service for API, and a one‑off Job.  
- **Hugging Face ecosystem** — Transformers/Datasets/Accelerate/PEFT. [9–12]

---

## Quickstart (Local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Windows: set a writeable artifacts directory
# PowerShell
mkdir data
$env:DATA_DIR = "$PWD/data"

uvicorn app:api --host 0.0.0.0 --port 8000 --reload
```

Submit a tiny demo job:
```bash
curl -X POST http://localhost:8000/distill -H "Content-Type: application/json" -d '{
  "teacher_model": "gpt2",
  "student_model": "distilgpt2",
  "dataset": "wikitext",
  "subset": "wikitext-2-raw-v1",
  "max_steps": 30,
  "per_device_batch_size": 2,
  "grad_accum_steps": 4,
  "logging_steps": 5,
  "save_every": 30,
  "temperature": 2.0
}'
```

List/download artifacts:
```bash
curl http://localhost:8000/artifacts/<job_id>
curl -L "http://localhost:8000/download/<job_id>/config.json" -o config.json
```

---

## Kubernetes (Demo)

```bash
# 1) Build & push
docker build -t <your-registry>/llm2slm:latest .
docker push <your-registry>/llm2slm:latest

# 2) Storage + defaults
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/config.yaml

# 3) API (edit image in k8s/api-deployment.yaml)
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml

# 4) One-off distillation Job (edit image in k8s/distill-job.yaml)
kubectl apply -f k8s/distill-job.yaml

# 5) Port-forward API
kubectl port-forward svc/llm2slm-api 8080:80
# POST to http://localhost:8080/distill
```

**GPU:** add resource requests/limits and the NVIDIA device plugin; the code auto‑uses CUDA if available.

---

## API

- `POST /distill` — start a distillation job (payload mirrors `schemas.DistillConfig`).  
- `GET /status/{job_id}` — return current/completed job status.  
- `GET /artifacts/{job_id}` — list generated files in the job folder.  
- `GET /download/{job_id}/{path}` — download any artifact.

**Example payload** (abridged):

```json
{
  "teacher_model": "gpt2",
  "student_model": "distilgpt2",
  "dataset": "wikitext",
  "subset": "wikitext-2-raw-v1",
  "split": "train",
  "max_steps": 200,
  "per_device_batch_size": 2,
  "grad_accum_steps": 8,
  "temperature": 2.0,
  "alpha_kl": 0.7,
  "beta_ce": 0.3,
  "gamma_mse": 0.05,
  "delta_cos": 0.05,
  "epsilon_attn": 0.0,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "use_qat": false,
  "post_training_quant": true,
  "export_onnx": true
}
```

---

## Algorithm (High‑Level)

```
Loss = α·KL(softmax(z_t/T) || softmax(z_s/T))·T^2
     + β·CE(z_s, argmax z_t)
     + γ·MSE(h_s, h_t)
     + δ·(1 − cos(h_s, h_t))
     + ε·MSE(A_s, A_t)            # optional attention alignment
```

- z_t, z_s: teacher/student logits; h: last hidden states; A: attention.  
- Temperature T smooths targets; LoRA optionally applied to student for efficiency. [5]  
- Post‑training dynamic quantization exports an `int8-dynamic/` variant; ONNX export is best‑effort. [6–8]

---

## Configuration & Paths

- **`DATA_DIR`**: base path for artifacts (default `/data`; use `./data` on Windows).  
- **`API_PREFIX`**: optional prefix for all routes (e.g., `/v1`).  
- **K8s ConfigMap** (`k8s/config.yaml`): default teacher/student/dataset and steps.  
- **Artifacts**: `DATA_DIR/artifacts/<job_id>/`. Contents include `config.json`, `pytorch_model.bin`, tokenizer files, `int8-dynamic/`, and (if supported) `model.onnx`.

---

## Known Good Dependency Set

Works on Windows (Python 3.13) and Linux:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
transformers>=4.45,<4.47
tokenizers>=0.20.1
datasets==3.0.1
accelerate==0.34.2
peft==0.13.2
torch>=2.2.0
onnx>=1.17.0
onnxruntime==1.23.0
numpy>=2.0
```

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531. <https://arxiv.org/abs/1503.02531>  
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108. <https://arxiv.org/abs/1910.01108>  
3. Jiao, X. et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding.* arXiv:1909.10351. <https://arxiv.org/abs/1909.10351>  
4. Romero, A. et al. (2015). *FitNets: Hints for Thin Deep Nets.* arXiv:1412.6550. <https://arxiv.org/abs/1412.6550>  
5. Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685. <https://arxiv.org/abs/2106.09685>  
6. Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR. arXiv:1712.05877. <https://arxiv.org/abs/1712.05877>  
7. ONNX – Open Neural Network Exchange. <https://onnx.ai/>  
8. ONNX Runtime Documentation. <https://onnxruntime.ai/docs/>  
9. Hugging Face Transformers Docs. <https://huggingface.co/docs/transformers>  
10. Hugging Face Datasets Docs. <https://huggingface.co/docs/datasets>  
11. Hugging Face PEFT Docs. <https://huggingface.co/docs/peft>  
12. Hugging Face Accelerate Docs. <https://huggingface.co/docs/accelerate>  
13. PyTorch Documentation. <https://pytorch.org/docs/stable/>  
14. Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. <https://arxiv.org/abs/1706.03762>

---

## License

Apache‑2.0 (or your preferred license).
