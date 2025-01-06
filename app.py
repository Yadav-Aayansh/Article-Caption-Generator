from flask import Flask, request, render_template, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the Hugging Face model and tokenizer
model_path = "t5-large"  # Replace with the path to your fine-tuned model directory
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

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
            article = "Generate a concise, engaging, and informative caption for the above news article. Use short forms, abbreviations, or acronyms where they are commonly recognized (e.g., 'Democratic National Committee' as 'DNC', '68000' as '68K', etc.). Ensure the caption captures the core essence of the article while maintaining brevity and readability." + article
            inputs = tokenizer.encode(article, return_tensors="pt", max_length=5012, truncation=True)
            outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(prediction)
            return render_template("prediction.html", prediction=prediction)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
