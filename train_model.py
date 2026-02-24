"""Fine-tune a legal document summarization model on local IN/UK datasets.

This script trains a LoRA adapter on the provided legal datasets under the
"dataset" directory. It uses train-data folders for training and test-data
folders for evaluation.
"""

import argparse
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
        # Device: config override or auto-detect
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
        """Convert row dicts into chat-style conversation examples.

        Each row must contain: Title, Abstract, DocumentText.
        """

        # Minimum content thresholds
        min_abs_len = 10
        min_doc_len = 100

        training_data = []
        for idx, row in enumerate(rows):
            title = str(row.get("Title", "")).strip()
            abstract = str(row.get("Abstract", "")).strip()
            document_text = str(row.get("DocumentText", "")).strip()

            if len(abstract) < min_abs_len or len(document_text) < min_doc_len:
                # Skip rows without enough signal for training
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
            summ_path = summary_dir / f.name
            abstract = self._read_text_file(summ_path)
            if not doc_text or not abstract:
                continue
            rows.append(
                {
                    "Title": f.stem,
                    "Abstract": abstract,
                    "DocumentText": doc_text,
                    "DatasetName": dataset_name,
                }
            )

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

            # Prefer full summary
            full_path = summary_root / "full" / f.name
            abstract = self._read_text_file(full_path)

            # Fall back to concatenating segment-wise summaries if needed
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

            rows.append(
                {
                    "Title": f.stem,
                    "Abstract": abstract,
                    "DocumentText": doc_text,
                    "DatasetName": dataset_name,
                }
            )

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
            summ_path = a1_full_dir / f.name
            abstract = self._read_text_file(summ_path)
            if not doc_text or not abstract:
                continue

            rows.append(
                {
                    "Title": f.stem,
                    "Abstract": abstract,
                    "DocumentText": doc_text,
                    "DatasetName": dataset_name,
                }
            )

        return rows

    def load_legal_datasets(self) -> None:
        """Load train/test rows from IN-Abs, UK-Abs, and IN-Ext datasets."""

        data_cfg = self.config.get("data", {})
        root = Path(data_cfg.get("root", "dataset"))
        max_train: Optional[int] = data_cfg.get("max_examples")
        max_test: Optional[int] = data_cfg.get("max_test_examples")
        use_in_abs = data_cfg.get("use_in_abs", True)
        use_uk_abs = data_cfg.get("use_uk_abs", True)
        use_in_ext = data_cfg.get("use_in_ext", True)
        train_rows: List[Dict] = []
        test_rows: List[Dict] = []

        # IN-Abs: abstractive Indian summaries
        if use_in_abs:
            in_abs_dir = root / "IN-Abs"
            in_abs_train = in_abs_dir / "train-data"
            in_abs_test = in_abs_dir / "test-data"
            train_rows.extend(self._load_in_abs_split(in_abs_train, "IN-Abs", max_train))
            test_rows.extend(self._load_in_abs_split(in_abs_test, "IN-Abs", max_test))

        # UK-Abs: abstractive UK summaries (full/segment-wise)
        if use_uk_abs:
            uk_abs_dir = root / "UK-Abs"
            uk_abs_train = uk_abs_dir / "train-data"
            uk_abs_test = uk_abs_dir / "test-data"
            train_rows.extend(self._load_uk_abs_split(uk_abs_train, "UK-Abs", max_train))
            test_rows.extend(self._load_uk_abs_split(uk_abs_test, "UK-Abs", max_test))

        # IN-Ext: extractive Indian summaries (A1 full only, train-set style)
        if use_in_ext:
            in_ext_dir = root / "IN-Ext"
            train_rows.extend(self._load_in_ext_train(in_ext_dir, "IN-Ext", max_train))

        if not train_rows:
            raise ValueError("No training rows loaded from legal datasets.")
        if not test_rows:
            print("[!] Warning: no test rows loaded from legal datasets.")

        self.train_rows = train_rows
        self.test_rows = test_rows

        print(f"[*] Loaded {len(train_rows)} training rows and {len(test_rows)} test rows from legal datasets.")

    # def setup_model_and_tokenizer(self):
    #     print("🤖 Setting up model and tokenizer...")
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
    #     if self.tokenizer.pad_token is None:
    #         self.tokenizer.pad_token = self.tokenizer.eos_token

    #     quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4"
    #     )

    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.model_name,
    #         quantization_config=quant_config,
    #         device_map="auto",
    #         trust_remote_code=True,
    #         torch_dtype=torch.float16
    #     )

    #     lora_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         inference_mode=False,
    #         r=16,
    #         lora_alpha=32,
    #         lora_dropout=0.1,
    #         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    #     )
    #     self.peft_model = get_peft_model(self.model, lora_config)
    #     self.peft_model.print_trainable_parameters()

    def setup_model_and_tokenizer(self):
        print("[*] Setting up model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side="right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ❌ Disable BitsAndBytes on Windows — not well supported locally
        # ✅ Use standard FP16 or FP32 load depending on GPU
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=None  # manually move to device
        ).to(self.device)

        # LoRA fine-tuning setup
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


    def prepare_training_data(self, training_data):
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
        return dataset

    def prepare_train_and_test_datasets(self):
        """Build Hugging Face Datasets for train and test splits."""

        train_examples = self._build_conversation_examples(self.train_rows)
        test_examples = self._build_conversation_examples(self.test_rows) if self.test_rows else []

        train_dataset = self.prepare_training_data(train_examples)
        test_dataset = self.prepare_training_data(test_examples) if test_examples else None

        return train_dataset, test_dataset

    def fine_tune_model(
        self,
        train_dataset,
        test_dataset=None,
        output_dir: Optional[str] = None,
    ):
        print("[*] Starting fine-tuning")
        train_cfg = self.config.get("training", {})
        if output_dir is None:
            output_dir = train_cfg.get("output_dir", "training_output")
        fp16_cfg = train_cfg.get("fp16", "auto")
        if fp16_cfg == "auto":
            use_fp16 = self.device == "cuda"
        else:
            use_fp16 = bool(fp16_cfg)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
            num_train_epochs=train_cfg.get("epochs", 3),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            fp16=use_fp16,
            logging_steps=train_cfg.get("logging_steps", 5),
            save_steps=train_cfg.get("save_steps", 50),
            save_total_limit=train_cfg.get("save_total_limit", 2),
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            dataloader_pin_memory=False,
        )

        def data_collator(examples):
            max_len = train_cfg.get("max_seq_length", 2048)
            batch = self.tokenizer(
                [ex["text"] for ex in examples],
                padding=True,
                truncation=True,
                max_length=max_len,
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
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"[+] Fine-tuning complete. Model saved to {output_dir}")

        # Save to pickle for Streamlit app
        with open("legal_summarizer.pkl", "wb") as f:
            pickle.dump((self.tokenizer, self.peft_model), f)
        print("[+] Pickle file saved: legal_summarizer.pkl")

        # Optional ROUGE evaluation on the test split
        eval_cfg = self.config.get("evaluation", {})
        if eval_cfg.get("enabled", True) and test_dataset is not None and self.test_rows:
            self.evaluate_on_test()

    def evaluate_on_test(self) -> None:
        """Generate summaries on the test set and report ROUGE scores."""

        if not self.test_rows:
            print("[!] No test rows available for evaluation.")
            return

        try:
            rouge = evaluate.load("rouge")
        except Exception as exc:  # pragma: no cover - evaluation is optional
            print(f"[!] Failed to load ROUGE metric: {exc}")
            return

        self.peft_model.eval()
        predictions: List[str] = []
        references: List[str] = []

        data_cfg = self.config.get("data", {})
        max_eval_examples: Optional[int] = data_cfg.get("max_test_examples")

        for idx, row in enumerate(self.test_rows):
            if max_eval_examples is not None and idx >= max_eval_examples:
                break

            title = str(row.get("Title", "")).strip()
            document_text = str(row.get("DocumentText", "")).strip()
            reference = str(row.get("Abstract", "")).strip()
            if not document_text or not reference:
                continue

            prompt = (
                "Summarize the following legal document in 2 paragraphs and provide key points.\n"
                f"Title: {title}\nDocument Content:\n{document_text}"
            )
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

            max_len = self.config.get("training", {}).get("max_seq_length", 2048)
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_cfg = self.config.get("evaluation", {}).get("generation", {})
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 256),
                    temperature=gen_cfg.get("temperature", 0.8),
                    top_p=gen_cfg.get("top_p", 0.9),
                    do_sample=gen_cfg.get("do_sample", True),
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "<|im_start|>assistant" in full_response:
                pred = full_response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
            else:
                pred = full_response.strip()

            if not pred:
                continue

            predictions.append(pred)
            references.append(reference)

        if not predictions:
            print("[!] No predictions generated for ROUGE evaluation.")
            return

        scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
        print("[*] ROUGE evaluation on test set:")
        for key, value in scores.items():
            try:
                print(f"  {key}: {value:.4f}")
            except TypeError:
                print(f"  {key}: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the legal summarizer using a YAML config file.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file (e.g., configs/Qwen1.5-0.5B-Chat_3.yaml)",
    )
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
    summarizer.fine_tune_model(
        train_dataset,
        test_dataset=test_dataset,
        output_dir=config.get("training", {}).get("output_dir"),
    )
