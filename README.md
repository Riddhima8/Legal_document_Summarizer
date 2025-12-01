
# ⚖️ Legal Document Summarizer — Finetuned LLM + LoRA + Streamlit

A production-grade legal summarization system powered by a fine-tuned variant of **Qwen1.5-0.5B-Chat**, trained using **LoRA** for domain-specific summarization. The project includes the full pipeline: dataset ingestion, model training, parameter-efficient fine-tuning, inference interface, and deployment.

## 🚀 Overview

Legal documents are lengthy, complex, and redundant. This project solves the problem using a lightweight fine-tuned large language model that produces accurate and readable summaries without requiring GPUs for inference.

It includes:

- Custom fine-tuning pipeline for legal summaries
- End-to-end model training & data preprocessing
- LoRA + PEFT adapter training
- Streamlit-based interactive UI
- Cloud deployment via Render

## 🧠 System Architecture

### ✔ Training
- Model: **Qwen/Qwen1.5-0.5B-Chat**
- Fine-Tuning: **LoRA + PEFT (Parameter Efficient Fine-Tuning)**
- Frameworks:
  - PyTorch
  - HuggingFace Transformers
  - HuggingFace Datasets
  - Pandas
- Training Data:
  - Legal abstracts from a Google Sheets CSV dataset
  - Converted into instruction format for chat LLMs

### ✔ Inference & Deployment
- Streamlit-based web interface
- Model serialization using `pickle`
- CPU-friendly inference pipeline
- Deployed using Render Web Services

## 🧰 Tech Stack & Why

### Core ML Components
| Tech | Reason |
|------|--------|
| **Qwen1.5-0.5B-Chat** | Tiny but powerful chat-tuned LLM |
| **Transformers** | Model loading, tokenization, generation |
| **PyTorch** | Training + GPU acceleration |
| **LoRA (PEFT)** | Efficient fine-tuning (90% fewer trainable params) |
| **Datasets** | Fast dataset creation and transformation |
| **BitsAndBytes** | (optional) 4-bit GPU loading |

### Backend & Application Side
| Technology | Reason |
|-----------|--------|
| **Streamlit** | UI without writing frontend code |
| **Render Cloud** | Easy automated deployment |
| **pickle** | Fast model loading without HF Hub |
| **Python 3.10** | Compatible dependency ecosystem |

This stack is optimized for:
✔ training small LLMs locally  
✔ deployment without GPUs  
✔ real-world inference throughput  

## 🗂 Project Structure

```
📦 Legal-Summarizer
│
├── train_model.py          # LoRA fine-tuning
├── streamlit_app.py        # Streamlit inference interface
├── legal_summarizer.pkl    # Saved tokenizer + finetuned model
├── requirements.txt
├── render.yaml             # Render deployment config
└── README.md
```

## 📌 Training Pipeline

### Step 1: Dataset Loading
Dataset is pulled from Google Sheets:

```
pd.read_csv("https://docs.google.com/.../export?format=csv")
```

### Step 2: Data Cleaning

- remove small/noisy records
- convert to conversational format required by Qwen:

```
<|im_start|>user ... <|im_end|>
<|im_start|>assistant ... <|im_end|>
```

### Step 3: LoRA Fine-Tuning

Example configuration:

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj","k_proj","v_proj",
        "o_proj","gate_proj","up_proj","down_proj"
    ]
)
```

This reduces training time & VRAM usage drastically.

## 🧪 Run Locally

### Install dependencies:

```
pip install -r requirements.txt
```

### Fine-tune the model:

```
python train_model.py
```

### Launch the Streamlit App:

```
streamlit run streamlit_app.py
```

## 🌐 Deployment

This project is deployed using **Render**.

### Start command:

```
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

### build command:

```
pip install -r requirements.txt
```

Deployment happens automatically from GitHub.

## 🖥 Streamlit UI

The interface allows:

- upload / paste legal document
- process & summarize using finetuned LLM
- display formatted summary output

This runs efficiently on CPU because only LoRA-adapted weights are loaded.

## 🔍 Example Usage

### Input
```
This agreement is entered into between the parties...
```

### Output
```
Summary:
- Defines legal obligations
- Specifies regulatory and compliance terms
- Outlines responsibilities of both parties
```

## 📈 Future Enhancements

- Chunking for very long documents
- Support for litigation, contracts, and compliance docs
- HuggingFace Spaces deployment
- Add ML evaluation metrics (ROUGE, BLEU, BERTScore)
- Support document uploads (PDF, DOCX)
- GPU-accelerated inference

## 🤝 Contributing

PRs and feedback welcome.  
Issues and feature requests are encouraged.

## ⭐ Final Note

This project demonstrates:

✔ Practical fine-tuning of LLMs  
✔ Real-world domain specialization  
✔ Efficient model deployment without GPU  
✔ Clean UI + backend integration  

A complete end-to-end pipeline for AI-powered legal summarization.
