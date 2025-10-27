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
    # Match the training format exactly: "Summarize:\nTitle: {title}\nDocument Content: {content}"
    # Extract title from first line of document
    lines = document_text.split('\n')
    title = lines[0] if lines else "Legal Document"
    
    # Format exactly like training data
    document_content = document_text[:3000]
    prompt = f"Summarize:\nTitle: {title}\nDocument Content: {document_content}"
    formatted_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(formatted_text, return_tensors="pt", truncation=True, max_length=3500).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2  # Prevent repetition
        )
    
    # Decode and extract only the assistant's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the assistant's part
    if "<|im_start|>assistant" in full_response:
        response = full_response.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
        # Clean up any remaining special tokens
        response = response.replace("<|im_end|>", "").strip()
        
        # If response still has the placeholder, try without the training format
        if "[Full text here]" in response or "Document Content:" in response:
            # Fallback: try simpler prompt
            simple_prompt = f"Summarize this document in detail:\n\n{document_text[:3000]}"
            simple_formatted = f"<|im_start|>user\n{simple_prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs2 = tokenizer(simple_formatted, return_tensors="pt", truncation=True, max_length=3500).to(device)
            with torch.no_grad():
                outputs2 = model.generate(**inputs2, max_new_tokens=800, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
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
