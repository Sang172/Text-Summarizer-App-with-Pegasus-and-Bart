from flask import Flask, request, render_template, jsonify
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Load models and tokenizers
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Define summarization functions
def summarize_with_pegasus(text):
    inputs = pegasus_tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = pegasus_model.generate(inputs["input_ids"], max_length=150, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)
    return pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_bart(text):
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Define routes
@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

@app.route("/summarize", methods=["POST"])
def summarize():
    text = request.form.get("text")
    model = request.form.get("model")

    if not text or not model:
        return jsonify({"error": "Text and model must be provided."})

    if model == "Pegasus":
        summary = summarize_with_pegasus(text)
    elif model == "BART":
        summary = summarize_with_bart(text)
    else:
        return jsonify({"error": "Invalid model selected."})

    return render_template('index.html', text=text, model=model, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)