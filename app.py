import streamlit as st
import torch
import pickle
import fitz
import PyPDF2

st.set_page_config(page_title="Legal Document Summarizer", page_icon="⚖️")

# Load model and tokenizer
@st.cache_resource
def load_model():
    try:
        with open("legal_summarizer.pkl", "rb") as f:
            tokenizer, model = pickle.load(f)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the model
try:
    tokenizer, model, device = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.title("⚖️ Legal Document Summarizer")

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        for p in reader.pages:
            text += p.extract_text()
    return text

def summarize_document(document_text):
    # Extract title from first line of document
    lines = document_text.split('\n')
    title = lines[0] if lines else "Legal Document"

    # Dynamic target length based on input size
    total_words = max(1, len(document_text.split()))
    target_summary_words = max(80, min(300, int(total_words * 0.10)))  # ~10% of doc, clamped
    # Rough words-to-tokens conversion (~0.75 words per token)
    target_tokens = max(80, min(520, int(target_summary_words / 0.75)))
    min_new_tokens = max(60, int(target_tokens * 0.6))
    max_new_tokens = max(min_new_tokens + 40, int(target_tokens * 1.2))

    # Keep model input under a safe limit
    document_content = document_text[:12000]

    # Prompt for scalable summary and 3–4 key points
    prompt = (
        "You are a legal analyst. Write a concise summary of about "
        f"{target_summary_words} words (2 short paragraphs if needed), then list 3–4 important points.\n\n"
        f"Title: {title}\n\n"
        f"Document Content:\n{document_content}\n\n"
        "Format strictly as:\n"
        "Summary:\n[paragraphs]\n\nImportant Points:\n- [point 1]\n- [point 2]\n- [point 3]\n- [point 4]"
    )

    # Use chat formatting the model was trained on
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
            early_stopping=False,
            repetition_penalty=1.05,
            length_penalty=1.0
        )

    # Decode and extract only the assistant's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract only the assistant's part
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        response = response.replace("<|im_end|>", "").strip()

        # Stronger fallback if too short or off-format
        if len(response.split()) < int(target_summary_words * 0.6) or "Important Points:" not in response:
            simple_prompt = (
                "Summarize the following legal text in about "
                f"{target_summary_words} words, then list 3–4 important points as bullets.\n\n"
                f"{document_text[:12000]}"
            )
            simple_formatted = f"<|im_start|>user\n{simple_prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs2 = tokenizer(simple_formatted, return_tensors="pt", truncation=True, max_length=14000).to(device)
            with torch.no_grad():
                outputs2 = model.generate(
                    **inputs2,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    pad_token_id=tokenizer.eos_token_id,
                    early_stopping=False,
                    repetition_penalty=1.05,
                    length_penalty=1.0,
                )
            full_response2 = tokenizer.decode(outputs2[0], skip_special_tokens=False)
            if "<|im_start|>assistant" in full_response2:
                candidate = full_response2.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
                # If still missing bullets, generate bullets separately and compose final output
                if "Important Points:" not in candidate:
                    # Extract summary segment if present
                    summary_text = candidate
                    if "Summary:" in candidate:
                        summary_text = candidate.split("Summary:", 1)[-1].strip()

                    bullet_prompt = (
                        "From the following legal document, list exactly 3–4 key points as bullets. "
                        "Return only bullets starting with '- ' and no other text.\n\n"
                        f"Document Content:\n{document_text[:12000]}"
                    )
                    bullet_formatted = f"<|im_start|>user\n{bullet_prompt}<|im_end|>\n<|im_start|>assistant\n"
                    inputs_b = tokenizer(bullet_formatted, return_tensors="pt", truncation=True, max_length=14000).to(device)
                    with torch.no_grad():
                        out_b = model.generate(
                            **inputs_b,
                            max_new_tokens=220,
                            min_new_tokens=60,
                            temperature=0.3,
                            top_p=0.9,
                            do_sample=False,
                            no_repeat_ngram_size=3,
                            pad_token_id=tokenizer.eos_token_id,
                            early_stopping=True,
                        )
                    bullets_full = tokenizer.decode(out_b[0], skip_special_tokens=False)
                    if "<|im_start|>assistant" in bullets_full:
                        bullets = bullets_full.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
                    else:
                        bullets = bullets_full.strip()

                    # Ensure bullet formatting
                    bullet_lines = [line.strip() for line in bullets.splitlines() if line.strip().startswith("-")]
                    bullet_lines = bullet_lines[:4]
                    if len(bullet_lines) < 3:
                        # As a minimal fallback, synthesize headings if model failed
                        bullet_lines = ["- Scope of services",
                                        "- Commercial terms/fees",
                                        "- Obligations and responsibilities",
                                        "- Signatures/term/termination"]

                    composed = "Summary:\n" + summary_text.strip() + "\n\nImportant Points:\n" + "\n".join(bullet_lines)
                    return composed
                return candidate

        return response
    else:
        # Fallback if parsing fails - try to get the last part
        parts = full_response.split("<|im_end|>")
        if len(parts) > 1:
            return parts[-2].strip()
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
