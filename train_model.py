"""Fine-tune a legal document summarization model on local IN/UK datasets.

This script trains a LoRA adapter on the provided legal datasets under the
"dataset" directory. It uses train-data folders for training and test-data
folders for evaluation.
"""

import argparse
import inspect
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")


class LegalDocumentSummarizerFast:
    """Fine-tunes a chat LLM on prepared legal-document examples."""

    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config["model"]["name"]
        cfg_device = str(config["model"].get("device", "auto")).lower()
        if cfg_device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else device
        elif cfg_device == "cpu":
            self.device = "cpu"
        else:
            self.device = device
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.train_rows = []
        self.test_rows = []

    @staticmethod
    def _build_conversation_examples(rows):
        """Convert row dicts into chat-style conversation examples."""

        min_abs_len = 10
        min_doc_len = 100
        training_data = []

        for idx, row in enumerate(rows):
            title = str(row.get("Title", "")).strip()
            abstract = str(row.get("Abstract", "")).strip()
            document_text = str(row.get("DocumentText", "")).strip()

            if len(abstract) < min_abs_len or len(document_text) < min_doc_len:
                continue

            training_data.append(
                {
                    "id": f"legal_doc_{idx}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": (
                                "Summarize the following legal document in 2 paragraphs and provide key points.\n"
                                f"Title: {title}\nDocument Content:\n{document_text}"
                            ),
                        },
                        {"from": "gpt", "value": f"**Summary:** {abstract}"},
                    ],
                }
            )

        if not training_data:
            raise ValueError("No valid rows found in dataset after filtering for content length.")

        return training_data

    @staticmethod
    def _read_text_file(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore").strip()
        except FileNotFoundError:
            return ""

    # ----------------- Dataset Loading -----------------
    def _load_in_abs_split(self, split_dir: Path, dataset_name: str, limit: Optional[int] = None):
        judgement_dir = split_dir / "judgement"
        summary_dir = split_dir / "summary"
        rows = []
        if not judgement_dir.is_dir() or not summary_dir.is_dir():
            return rows

        files = sorted([p for p in judgement_dir.glob("*.txt")])
        if limit is not None:
            files = files[:limit]

        for f in files:
            doc_text = self._read_text_file(f)
            abstract = self._read_text_file(summary_dir / f.name)
            if not doc_text or not abstract:
                continue
            rows.append({"Title": f.stem, "Abstract": abstract, "DocumentText": doc_text, "DatasetName": dataset_name})
        return rows

    def _load_uk_abs_split(self, split_dir: Path, dataset_name: str, limit: Optional[int] = None):
        judgement_dir = split_dir / "judgement"
        summary_root = split_dir / "summary"
        rows = []
        if not judgement_dir.is_dir() or not summary_root.is_dir():
            return rows

        files = sorted([p for p in judgement_dir.glob("*.txt")])
        if limit is not None:
            files = files[:limit]

        for f in files:
            doc_text = self._read_text_file(f)
            if not doc_text:
                continue

            full_path = summary_root / "full" / f.name
            abstract = self._read_text_file(full_path)

            if not abstract:
                seg_root = summary_root / "segment-wise"
                segments = []
                for seg_name in ["background", "judgement", "reasons"]:
                    seg_path = seg_root / seg_name / f.name
                    seg_text = self._read_text_file(seg_path)
                    if seg_text:
                        segments.append(seg_text)
                abstract = "\n\n".join(segments).strip()

            if not abstract:
                continue

            rows.append({"Title": f.stem, "Abstract": abstract, "DocumentText": doc_text, "DatasetName": dataset_name})
        return rows

    def _load_in_ext_train(self, root_dir: Path, dataset_name: str, limit: Optional[int] = None):
        judgement_dir = root_dir / "judgement"
        a1_full_dir = root_dir / "summary" / "full" / "A1"
        rows = []
        if not judgement_dir.is_dir() or not a1_full_dir.is_dir():
            return rows

        files = sorted([p for p in judgement_dir.glob("*.txt")])
        if limit is not None:
            files = files[:limit]

        for f in files:
            doc_text = self._read_text_file(f)
            abstract = self._read_text_file(a1_full_dir / f.name)
            if not doc_text or not abstract:
                continue
            rows.append({"Title": f.stem, "Abstract": abstract, "DocumentText": doc_text, "DatasetName": dataset_name})
        return rows

    def load_legal_datasets(self) -> None:
        """Load train/test rows from IN-Abs, UK-Abs, and IN-Ext datasets."""
        data_cfg = self.config.get("data", {})
        root = Path(data_cfg.get("root", "dataset"))
        max_train: Optional[int] = data_cfg.get("max_examples")
        max_test: Optional[int] = data_cfg.get("max_test_examples")

        train_rows: List[Dict] = []
        test_rows: List[Dict] = []

        if data_cfg.get("use_in_abs", True):
            in_abs_dir = root / "IN-Abs"
            train_rows.extend(self._load_in_abs_split(in_abs_dir / "train-data", "IN-Abs", max_train))
            test_rows.extend(self._load_in_abs_split(in_abs_dir / "test-data", "IN-Abs", max_test))

        if data_cfg.get("use_uk_abs", True):
            uk_abs_dir = root / "UK-Abs"
            train_rows.extend(self._load_uk_abs_split(uk_abs_dir / "train-data", "UK-Abs", max_train))
            test_rows.extend(self._load_uk_abs_split(uk_abs_dir / "test-data", "UK-Abs", max_test))

        if data_cfg.get("use_in_ext", True):
            in_ext_dir = root / "IN-Ext"
            train_rows.extend(self._load_in_ext_train(in_ext_dir, "IN-Ext", max_train))

        if not train_rows:
            raise ValueError("No training rows loaded from legal datasets.")
        if not test_rows:
            print("[!] Warning: no test rows loaded from legal datasets.")

        self.train_rows = train_rows
        self.test_rows = test_rows
        print(f"[*] Loaded {len(train_rows)} training rows and {len(test_rows)} test rows from legal datasets.")

    # ----------------- Model Setup -----------------
    def setup_model_and_tokenizer(self):
        print("[*] Setting up model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float32

        # For training, avoid model sharding/offload (`device_map="auto"`) because
        # it can leave parts on meta/offloaded devices and break backward with PEFT.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        self.model.to(self.device)

        lora_cfg = self.config.get("lora", {})
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()

    # ----------------- Dataset Preparation -----------------
    def prepare_training_data(self, training_data):
        """Pre-tokenize dataset to avoid repeated tokenization in the collator."""
        def format_conv(example):
            text = ""
            for conv in example["conversations"]:
                if conv["from"] == "human":
                    text += f"<|im_start|>user\n{conv['value']}<|im_end|>\n"
                else:
                    text += f"<|im_start|>assistant\n{conv['value']}<|im_end|>\n"
            return {"text": text}

        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(format_conv)

        max_len = self.config.get("training", {}).get("max_seq_length", 512)
        def tokenize(example):
            return self.tokenizer(
                example["text"],
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )

        dataset = dataset.map(tokenize, batched=True, batch_size=4)
        return dataset

    def prepare_train_and_test_datasets(self):
        train_examples = self._build_conversation_examples(self.train_rows)
        test_examples = self._build_conversation_examples(self.test_rows) if self.test_rows else []

        train_dataset = self.prepare_training_data(train_examples)
        test_dataset = self.prepare_training_data(test_examples) if test_examples else None
        return train_dataset, test_dataset

    # ----------------- Fine-tuning -----------------
    # ----------------- Fine-tuning -----------------
    def fine_tune_model(self, train_dataset, test_dataset=None, output_dir: Optional[str] = None):
        print("[*] Starting fine-tuning")
        train_cfg = self.config.get("training", {})
        if output_dir is None:
            output_dir = train_cfg.get("output_dir", "training_output")
        use_fp16 = (self.device == "cuda") if train_cfg.get("fp16", "auto") == "auto" else bool(train_cfg.get("fp16"))

        training_args_kwargs = {
            "output_dir": output_dir,
            "per_device_train_batch_size": train_cfg.get("per_device_train_batch_size", 1),
            "gradient_accumulation_steps": train_cfg.get("gradient_accumulation_steps", 1),
            "num_train_epochs": train_cfg.get("epochs", 3),
            "learning_rate": train_cfg.get("learning_rate", 2e-4),
            "fp16": use_fp16,
            "logging_steps": train_cfg.get("logging_steps", 5),
            "save_steps": train_cfg.get("save_steps", 50),
            "save_total_limit": train_cfg.get("save_total_limit", 2),
            "remove_unused_columns": False,
            "report_to": "none",
            "dataloader_pin_memory": False,
            "max_grad_norm": 1.0,
        }

        # Keep compatibility across transformers versions.
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if self.device == "cpu":
            if "no_cuda" in ta_params:
                training_args_kwargs["no_cuda"] = True
            elif "use_cpu" in ta_params:
                training_args_kwargs["use_cpu"] = True

        training_args = TrainingArguments(**training_args_kwargs)

        # Updated data collator
        def data_collator(examples):
            batch = self.tokenizer(
                [ex["text"] for ex in examples],
                padding=True,
                truncation=True,  # ⚠️ important to avoid grad_norm=nan
                max_length=train_cfg.get("max_seq_length", 512),
                return_tensors="pt",
            )
            batch["labels"] = batch["input_ids"].clone()
            return batch

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )

        trainer.train()

        # Save model & tokenizer correctly
        trainer.save_model(output_dir)
        self.peft_model.save_pretrained(output_dir)  # ⚡ save LoRA adapter safely
        self.tokenizer.save_pretrained(output_dir)
        print(f"[+] Fine-tuning complete. Model saved to {output_dir}")

        print("[+] Model saved successfully. Pickle is no longer needed for PEFT models.")

    # ----------------- Evaluation -----------------
    def evaluate_on_test(self) -> None:
        if not self.test_rows:
            print("[!] No test rows available for evaluation.")
            return

        rouge = evaluate.load("rouge")
        self.peft_model.eval()
        predictions: List[str] = []
        references: List[str] = []

        max_eval_examples: Optional[int] = self.config.get("data", {}).get("max_test_examples")

        for idx, row in enumerate(self.test_rows):
            if max_eval_examples is not None and idx >= max_eval_examples:
                break

            title = row["Title"].strip()
            document_text = row["DocumentText"].strip()
            reference = row["Abstract"].strip()
            if not document_text or not reference:
                continue

            prompt = f"Summarize the following legal document in 2 paragraphs and provide key points.\nTitle: {title}\nDocument Content:\n{document_text}"
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True,
                                    max_length=self.config.get("training", {}).get("max_seq_length", 512))
            model_device = next(self.peft_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=self.config.get("evaluation", {}).get("generation", {}).get("max_new_tokens", 256),
                    temperature=self.config.get("evaluation", {}).get("generation", {}).get("temperature", 0.8),
                    top_p=self.config.get("evaluation", {}).get("generation", {}).get("top_p", 0.9),
                    do_sample=self.config.get("evaluation", {}).get("generation", {}).get("do_sample", True),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            pred = full_response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            predictions.append(pred)
            references.append(reference)

        scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        print("[*] ROUGE evaluation on test set:")
        for key, value in scores.items():
            print(f"  {key}: {value:.4f}")


# ----------------- Main -----------------
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the legal summarizer using a YAML config file.")
    parser.add_argument("--config", required=True,
                        help="Path to the YAML config file (e.g., configs/Qwen1.5-0.5B-Chat_3.yaml)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summarizer = LegalDocumentSummarizerFast(config=config)
    summarizer.setup_model_and_tokenizer()
    summarizer.load_legal_datasets()
    train_dataset, test_dataset = summarizer.prepare_train_and_test_datasets()
    summarizer.fine_tune_model(train_dataset, test_dataset=test_dataset,
                               output_dir=config.get("training", {}).get("output_dir"))