import streamlit as st
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer

# Load Pegasus model and tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# Load BART model and tokenizer
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def summarize_with_pegasus(text):
    inputs = pegasus_tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = pegasus_model.generate(inputs["input_ids"], max_length=150, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_bart(text):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit app
st.title("Text Summarizer with Pegasus and BART")

# Dropdown menu for model selection
model_option = st.selectbox(
    "Select Summarization Model",
    ["Pegasus", "BART"]
)

text = st.text_area("Enter the text to summarize",height=500)

if st.button("Summarize"):
    if text:
        if model_option == "Pegasus":
            summary = summarize_with_pegasus(text)
            st.subheader("Pegasus Summary")
        elif model_option == "BART":
            summary = summarize_with_bart(text)
            st.subheader("BART Summary")
        st.write(summary)
    else:
        st.error("Please enter some text to summarize.")