from flask import Flask, request, jsonify
import joblib
from utils import clean_text, ensure_nltk
import traceback

ensure_nltk()
app = Flask(__name__)
# load saved model
obj = joblib.load("model.joblib")
model = obj["model"]
vectorizer = obj["vectorizer"]

@app.route("/")
def home():
    return "Fake News Detection API. POST /predict with JSON {'text': '...'}"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text","") if isinstance(data, dict) else ""
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X).max()
        label = "real" if pred==1 else "fake"
        return jsonify({"label": label, "confidence": float(proba)})
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

if __name__ == "__main__":
    app.run(debug=True)