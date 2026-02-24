# train_model.py
import argparse
import pickle
import warnings

import pandas as pd
import torch
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

    def __init__(self, model_name="Qwen/Qwen1.5-0.5B-Chat"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self.peft_model = None

    def load_training_examples(self, csv_path: str, max_examples: int = 50):
        """Load pre-processed rows from disk and build chat-style conversations."""

        print(f"[*] Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        df = df.head(max_examples)

        required_columns = {"Title", "Abstract", "DocumentText"}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Dataset must contain columns: Title, Abstract, DocumentText. "
                f"Missing: {', '.join(sorted(missing_columns))}"
            )

        training_data = []
        for idx, row in df.iterrows():
            title = str(row.get("Title", "")).strip()
            abstract = str(row.get("Abstract", "")).strip()
            document_text = str(row.get("DocumentText", "")).strip()

            if len(abstract) < 10 or len(document_text) < 100:
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

        print(f"Loaded {len(training_data)} usable examples for training")
        return training_data

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
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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

    def fine_tune_model(self, dataset, output_dir="./legal_summarizer_fast"):
        print("[*] Starting fine-tuning (3 epochs)...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=5,
            save_steps=50,
            eval_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            dataloader_pin_memory=False,
        )

        def data_collator(examples):
            batch = self.tokenizer([ex["text"] for ex in examples],
                                   padding=True, truncation=True, max_length=2048, return_tensors="pt")
            batch["labels"] = batch["input_ids"].clone()
            return batch

        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune the legal summarizer on a prepared CSV dataset.")
    parser.add_argument(
        "--dataset_csv",
        required=True,
        help="Path to the preprocessed CSV containing Title, Abstract, DocumentText columns.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=50,
        help="Upper bound on number of rows to use from the dataset (default: 50).",
    )
    parser.add_argument(
        "--output_dir",
        default="./legal_summarizer_fast",
        help="Directory where the fine-tuned model and tokenizer will be stored.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    summarizer = LegalDocumentSummarizerFast()
    training_data = summarizer.load_training_examples(args.dataset_csv, max_examples=args.max_examples)
    summarizer.setup_model_and_tokenizer()
    dataset = summarizer.prepare_training_data(training_data)
    summarizer.fine_tune_model(dataset, output_dir=args.output_dir)
