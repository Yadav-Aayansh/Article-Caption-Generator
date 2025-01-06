from flask import Flask, request, render_template, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gdown
import os

app = Flask(__name__)

# Path to save the downloaded model
MODEL_PATH = "fine_tuned_t5"

# Download the model from Google Drive if it doesn't exist locally
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your Google Drive file ID
    zip_path = "model.zip"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)
    
    # Extract the zip file
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MODEL_PATH)
    os.remove(zip_path)  # Clean up zip file
    print("Model downloaded and extracted successfully.")

# Load the Hugging Face model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

print("Model and tokenizer loaded successfully.")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")  # Render the HTML for the homepage
    else:
        # Handle POST request to get predictions
        article = request.form.get('article')
        if not article:
            return jsonify({"error": "No article provided"}), 400

        try:
            # Prepend instructions to the article
            article = (
                "Generate a concise, engaging, and informative caption for the above news article. "
                "Use short forms, abbreviations, or acronyms where they are commonly recognized "
                "(e.g., 'Democratic National Committee' as 'DNC', '68000' as '68K', etc.). "
                "Ensure the caption captures the core essence of the article while maintaining brevity and readability. "
                + article
            )
            inputs = tokenizer.encode(article, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(prediction)
            return render_template("prediction.html", prediction=prediction)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
