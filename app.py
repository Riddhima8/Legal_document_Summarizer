import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import pypdfium2 as pdfium

st.set_page_config(page_title="Legal Document Summarizer", page_icon="⚖️")

MODEL_PATH = "training_output"  # Your saved PEFT model folder
BASE_MODEL_NAME = "Qwen/Qwen1.5-0.5B-Chat"  # e.g., Qwen-1.5B-Chat

# Load model and tokenizer
@st.cache_resource
def load_model():
    # 1. Don't manually define 'device' for the model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # 2. Load the base model with auto mapping
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",  # This handles the heavy lifting
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )

        # 3. Load the LoRA adapter
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        # ❌ REMOVE THIS LINE: model.to(device) 
        # It conflicts with device_map="auto"
        
        model.eval()
        
        # 4. We still need to know where the 'inputs' go
        # We'll grab the device from the first model parameter
        current_device = next(model.parameters()).device
        
        return tokenizer, model, current_device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
tokenizer, model, device = load_model()

st.title("⚖️ Legal Document Summarizer")

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        pdf = pdfium.PdfDocument(uploaded_file)
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            text_page = page.get_textpage()
            text += text_page.get_text_range() + "\n"
            text_page.close()
            page.close()
        pdf.close()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
    return text

def summarize_document(document_text):
    lines = document_text.split('\n')
    title = lines[0] if lines else "Legal Document"

    total_words = max(1, len(document_text.split()))
    target_summary_words = max(80, min(300, int(total_words * 0.10)))
    target_tokens = max(80, min(520, int(target_summary_words / 0.75)))
    min_new_tokens = max(60, int(target_tokens * 0.6))
    max_new_tokens = max(min_new_tokens + 40, int(target_tokens * 1.2))

    document_content = document_text[:12000]
    prompt = (
        "You are a legal analyst. Write a concise summary of about "
        f"{target_summary_words} words (2 short paragraphs if needed), then list 3–4 important points.\n\n"
        f"Title: {title}\n\n"
        f"Document Content:\n{document_content}\n\n"
        "Format strictly as:\n"
        "Summary:\n[paragraphs]\n\nImportant Points:\n- [point 1]\n- [point 2]\n- [point 3]\n- [point 4]"
    )

    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=14000).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        return response
    return full_response

uploaded_pdf = st.file_uploader("📄 Upload a Legal PDF", type=["pdf"])

if uploaded_pdf:
    pdf_text = extract_text_from_pdf(uploaded_pdf)
    st.success("✅ PDF text extracted.")
    if st.button("Generate Summary"):
        with st.spinner("Summarizing and extracting key points..."):
            result = summarize_document(pdf_text)
            st.subheader("📜 Document Summary & Key Points")
            st.write(result)
            st.download_button("💾 Download Summary", result, file_name="summary.txt")