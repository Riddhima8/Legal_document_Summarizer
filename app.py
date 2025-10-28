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

    # Encourage a half-page summary and explicit key points
    document_content = document_text[:5000]
    prompt = (
        "You are a legal analyst. Write a detailed, half‑page summary (200–300 words) "
        "in 2–3 short paragraphs, followed by a clear bulleted list of 6–10 important points.\n\n"
        f"Title: {title}\n\n"
        f"Document Content:\n{document_content}\n\n"
        "Format strictly as:\n"
        "Summary:\n[paragraphs]\n\nImportant Points:\n- [point 1]\n- [point 2]\n- [point 3]"
    )

    # Use chat formatting the model was trained on
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=5500).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=700,
            min_new_tokens=250,  # Ensure half-page length
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
        if len(response.split()) < 180 or "Important Points:" not in response:
            simple_prompt = (
                "Summarize the following legal text in 200–300 words (2–3 paragraphs) "
                "and then list 6–10 important points as bullets.\n\n"
                f"{document_text[:5000]}"
            )
            simple_formatted = f"<|im_start|>user\n{simple_prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs2 = tokenizer(simple_formatted, return_tensors="pt", truncation=True, max_length=5500).to(device)
            with torch.no_grad():
                outputs2 = model.generate(
                    **inputs2,
                    max_new_tokens=700,
                    min_new_tokens=250,
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
                return full_response2.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()

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
