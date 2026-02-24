
# ⚖️ Legal Document Summarizer

End-to-end system for summarizing long legal case documents using a fine-tuned **Qwen1.5-0.5B-Chat** model with **LoRA**. The project includes:

- Local legal datasets (Indian & UK Supreme Court)
- Config-driven training pipeline
- Parameter-efficient fine-tuning (PEFT/LoRA)
- ROUGE-based evaluation
- Streamlit web app for interactive summarization

---

## 1. Overview

Legal judgements are long, repetitive, and hard to skim. This project fine-tunes a small chat LLM on real legal case datasets so it can produce concise, structured summaries and key points for new legal documents.

Key properties:

- Runs on modest hardware (CPU or single GPU)
- Uses LoRA adapters instead of full-model finetuning
- Reads curated legal datasets from disk (no external API dependency)
- Training and evaluation are controlled via YAML configs

---

## 2. Project Structure

```text
Legal_document_Summarizer/
├── app.py                      # Streamlit UI for inference
├── train_model.py              # Training & evaluation entrypoint
├── requirements.txt            # Python dependencies
├── configs/
│   └── Qwen1.5-0.5B-Chat_3.yaml  # Example training config
├── dataset/                    # Legal datasets (IN-Abs, IN-Ext, UK-Abs)
│   ├── DATASET_README.md
│   ├── IN-Abs/
│   ├── IN-Ext/
│   └── UK-Abs/
├── legal_summarizer_fast/      # Saved adapter + tokenizer (created after training)
├── legal_summarizer.pkl        # Pickled tokenizer + adapter (created after training)
└── README.md
```

---

## 3. Datasets

All datasets live under the `dataset/` directory and come from the paper:

> *Legal Case Document Summarization: Extractive and Abstractive Methods and their Evaluation* (AACL-IJCNLP 2022)

See `dataset/DATASET_README.md` for full details. In short:

### 3.1 IN-Abs (Indian, abstractive)

- Source: Indian Supreme Court cases (INSC)
- Structure:
  - `IN-Abs/train-data/judgement/` — full case texts (train)
  - `IN-Abs/train-data/summary/`   — abstractive summaries (train)
  - `IN-Abs/test-data/judgement/`  — full case texts (test)
  - `IN-Abs/test-data/summary/`    — abstractive summaries (test)
- Usage in this project:
  - Train split → training
  - Test split  → evaluation

### 3.2 UK-Abs (UK, abstractive + segment-wise)

- Source: UK Supreme Court decisions
- Structure (mirrors IN-Abs) with additional segment-wise summaries:
  - `UK-Abs/train-data/judgement/`
  - `UK-Abs/train-data/summary/full/` (if present)
  - `UK-Abs/train-data/summary/segment-wise/{background,judgement,reasons}/`
  - Same layout under `test-data/`
- Usage in this project:
  - For each judgement, prefer `summary/full/`. If missing, concatenate `background`, `judgement`, `reasons` segments.
  - Train split → training
  - Test split  → evaluation

### 3.3 IN-Ext (Indian, extractive)

- 50 Indian Supreme Court cases with human-written extractive summaries.
- Structure:
  - `IN-Ext/judgement/`           — full case texts
  - `IN-Ext/summary/full/A1/`     — "full" extractive summaries by annotator A1
  - additional folders for A2 + segment-wise labels (not used here)
- Usage in this project:
  - Only `summary/full/A1` is used, as **extra training data** (no test split).

---

## 4. Configuration (YAML)

Training is driven entirely by a YAML config file stored in `configs/`. The main entrypoint is `train_model.py`, which only accepts:

```bash
python train_model.py --config configs/Qwen1.5-0.5B-Chat_3.yaml
```

### 4.1 Example config (`configs/Qwen1.5-0.5B-Chat_3.yaml`)

```yaml
model:
  name: "Qwen/Qwen1.5-0.5B-Chat"
  device: "auto"  # "auto" | "cuda" | "cpu"

lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

data:
  root: "dataset"
  use_in_abs: true
  use_uk_abs: true
  use_in_ext: true
  max_examples: null        # per train split; null = use all
  max_test_examples: null   # per test split / ROUGE; null = use all
  min_abstract_length: 10
  min_document_length: 100

training:
  output_dir: "training_output"
  epochs: 3
  learning_rate: 2.0e-4
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  max_seq_length: 2048
  logging_steps: 5
  save_steps: 50
  save_total_limit: 2
  fp16: "auto"   # "auto" | true | false

evaluation:
  enabled: true
  metric: "rouge"
  generation:
    max_new_tokens: 256
    temperature: 0.8
    top_p: 0.9
    do_sample: true
```

You can create new configs with the naming pattern:

```text
configs/[ModelName]_[Epochs].yaml
```

and then pass them via `--config`.

---

## 5. Training Pipeline

The main training logic lives in `train_model.py` (class `LegalDocumentSummarizerFast`).

### 5.1 Data loading

- Reads `data.root` from config (`dataset/` by default).
- Conditionally loads datasets based on `use_in_abs`, `use_uk_abs`, `use_in_ext`.
- For each judgement file, finds the aligned summary (as described in the dataset section).
- Applies basic filters:
  - `min_abstract_length` (default 10 characters)
  - `min_document_length` (default 100 characters)

The result is **train_rows** and **test_rows**:

```python
{
  "Title": "case_id",
  "Abstract": "reference summary text",
  "DocumentText": "full case text",
  "DatasetName": "IN-Abs" | "UK-Abs" | "IN-Ext",
}
```

### 5.2 Conversational formatting

Each (document, summary) pair is converted into a chat-style example:

- User message:

  ```text
  Summarize the following legal document in 2 paragraphs and provide key points.
  Title: <title>
  Document Content:
  <full judgement text>
  ```

- Assistant message:

  ```text
  **Summary:** <reference summary>
  ```

These are then wrapped with Qwen’s chat markers:

```text
<|im_start|>user
...prompt...
<|im_end|>
<|im_start|>assistant
...summary...
<|im_end|>
```

and stored as a `text` field in a Hugging Face `Dataset`.

### 5.3 Model & LoRA setup

- Loads `model.name` from config via `AutoModelForCausalLM` and `AutoTokenizer`.
- Sets `pad_token` to `eos_token` if needed.
- Uses FP16 on CUDA and FP32 on CPU by default (overridable via `training.fp16`).
- Applies LoRA with parameters from `lora.*`.

### 5.4 Fine-tuning

Training hyperparameters are taken from the `training` section of the config:

- `epochs`, `learning_rate`, `per_device_train_batch_size`
- `gradient_accumulation_steps`, `max_seq_length`
- `logging_steps`, `save_steps`, `save_total_limit`

The script uses Hugging Face `Trainer` to:

- Train on the prepared `train_dataset`.
- Optionally attach a `test_dataset` for loss-based evaluation (if enabled in the future).
- Save the LoRA adapter and tokenizer to `training.output_dir`.
- Save a pickled `(tokenizer, peft_model)` tuple to `legal_summarizer.pkl` for the app.

### 5.5 Evaluation (ROUGE)

If `evaluation.enabled: true`, the script:

- Generates summaries on the test rows from IN-Abs and UK-Abs.
- Uses generation settings from `evaluation.generation`.
- Computes ROUGE scores using the `evaluate` library:
  - `rouge1`, `rouge2`, `rougeL`, `rougeLsum`.
- Prints aggregate scores to the console.

The number of evaluated examples is capped by `data.max_test_examples` (or all if `null`).

---

## 6. Running Locally

### 6.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 6.2 Fine-tune the model

Using the provided example config:

```bash
python train_model.py --config configs/Qwen1.5-0.5B-Chat_3.yaml
```

To customize training (e.g., epochs, learning rate, max_examples), copy that YAML, tweak values, and point `--config` to your new file.

After training completes, you should see:

- A trained adapter and tokenizer in `training_output/` (or your configured `output_dir`).
- A `legal_summarizer.pkl` file in the project root.

---

## 7. Streamlit App (Inference)

The Streamlit app in `app.py`:

- Loads `legal_summarizer.pkl` (tokenizer + LoRA model).
- Accepts a PDF upload or pasted text.
- Extracts text (via `pypdfium2` for PDFs).
- Builds a summarization prompt and calls `model.generate`.
- Displays a structured output with a summary + important points.

### 7.1 Run the app locally

```bash
streamlit run app.py
```

Open the URL printed by Streamlit in your browser to use the UI.

---

## 8. Deployment (optional)

You can deploy the Streamlit app to platforms like Render or similar PaaS.

Example start command for a container or Render service:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Build command:

```bash
pip install -r requirements.txt
```

Make sure the trained adapter directory and `legal_summarizer.pkl` are included in the deployment image.

---

## 9. Tech Stack

- **Model**: Qwen/Qwen1.5-0.5B-Chat
- **Training**: PyTorch, Hugging Face Transformers, Datasets, PEFT (LoRA)
- **Metrics**: ROUGE via `evaluate`, `rouge_score`, `nltk`
- **App**: Streamlit
- **PDF parsing**: pypdfium2

---

## 10. Future Enhancements

- Chunking for very long documents
- Per-dataset metrics (IN-Abs vs UK-Abs vs IN-Ext)
- Additional evaluation metrics (BLEU, BERTScore)
- Support for contracts and other legal document types
- More inference controls (temperature, length sliders) in the UI

---

## 11. Contributing

Suggestions and improvements are welcome. If you see a bug or have an idea:

- Open an issue describing the problem or feature.
- Or submit a pull request with a clear description of the change.

---

## 12. Summary

This repository provides a complete, reproducible pipeline for:

- Preparing legal summarization data from established research datasets
- Fine-tuning a small chat LLM with LoRA on legal cases
- Evaluating with ROUGE
- Serving the model via a simple but effective Streamlit UI

It’s a practical starting point for real-world, domain-specific summarization systems in the legal domain.
