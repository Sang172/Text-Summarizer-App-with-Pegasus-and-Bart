from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, BartForConditionalGeneration, BartTokenizer

# Initialize FastAPI app
app = FastAPI()

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

# Define API request models
class SummarizeRequest(BaseModel):
    text: str
    model: str

# Define API endpoints
@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    if request.model == "Pegasus":
        summary = summarize_with_pegasus(request.text)
    elif request.model == "BART":
        summary = summarize_with_bart(request.text)
    else:
        return {"error": "Invalid model selected."}
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <style>
                body {
                    background-color: #e6f7ff; /* Light blue background */
                    font-family: Arial, sans-serif; /* Easy-to-read font */
                    text-align: center; /* Center text */
                    margin: 0;
                    padding: 0;
                }
                h1 {
                    color: #0056b3; /* Dark blue color for the header */
                }
                form {
                    display: inline-block;
                    background-color: #ffffff; /* White background for the form */
                    padding: 20px;
                    border-radius: 8px; /* Rounded corners */
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Subtle shadow */
                    width: 80%; /* Set form width to 80% of the viewport width */
                    max-width: 800px; /* Maximum width of the form */
                    box-sizing: border-box; /* Include padding in width calculation */
                }
                textarea {
                    width: 100%;
                    border-radius: 4px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    box-sizing: border-box; /* Include padding in width calculation */
                }
                select, button {
                    border-radius: 4px;
                    padding: 10px;
                    border: 1px solid #ccc;
                    margin-top: 10px;
                    box-sizing: border-box; /* Include padding in width calculation */
                }
                button {
                    background-color: #0056b3;
                    color: white;
                    cursor: pointer;
                    border: none;
                    padding: 10px 20px;
                }
                button:hover {
                    background-color: #004494;
                }
                #result {
                    margin-top: 20px;
                    padding: 20px;
                    background-color: #ffffff; /* White background for the result box */
                    border-radius: 8px; /* Rounded corners */
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Subtle shadow */
                    width: 80%; /* Set width to 80% of the viewport width */
                    max-width: 800px; /* Maximum width of the result box */
                    text-align: left; /* Align text to the left */
                    box-sizing: border-box; /* Include padding in width calculation */
                    margin: 20px auto; /* Center the result box */
                }
            </style>
        </head>
        <body>
            <h1>Text Summarizer with Pegasus and BART</h1>
            <form id="summarizeForm">
                <label for="text">Enter the text to summarize:</label><br>
                <textarea id="text" name="text" rows="10" cols="50"></textarea><br><br>
                <label for="model">Select Summarization Model:</label><br>
                <select id="model" name="model">
                    <option value="Pegasus">Pegasus</option>
                    <option value="BART">BART</option>
                </select><br><br>
                <button type="button" onclick="submitForm()">Summarize</button>
            </form>
            <div id="result"></div>

            <script>
                async function submitForm() {
                    const text = document.getElementById('text').value;
                    const model = document.getElementById('model').value;
                    const response = await fetch('/summarize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text, model: model }),
                    });
                    const data = await response.json();
                    document.getElementById('result').innerText = 'Summary: ' + data.summary;
                }
            </script>
        </body>
    </html>
    """
    return html_content
