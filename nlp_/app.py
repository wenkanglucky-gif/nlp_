from flask import Flask, request, jsonify
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from datetime import datetime
import os
import pandas as pd
import csv

app = Flask(__name__)

#  Load model
model_path = "saved_legalbert"
model = TFAutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Monitoring Setup
MONITOR_DIR = "monitor_logs"
ALERTS_FILE = os.path.join(MONITOR_DIR, "alerts.log")
os.makedirs(MONITOR_DIR, exist_ok=True)


def log_model_metrics(pred_label, confidence, text):
    """
    Logs prediction metrics to CSV file.
    """

    log_file = os.path.join(MONITOR_DIR, "metrics.csv")

    # Write header if file doesn't exist
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "prediction", "confidence", "input_length"])
        writer.writerow(
            [
                datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                int(pred_label),
                float(confidence),
                len(text),
            ]
        )

    print(f"Logged metrics: pred={pred_label}, conf={confidence:.3f}")


def check_alert(confidence):
    """
    Simple warning system for low confidence predictions.
    """
    if confidence < 0.50:
        alert_msg = f"[{datetime.utcnow()}] ALERT: Low confidence {confidence:.3f}\n"
        print(alert_msg)
        # Log to file
        with open(ALERTS_FILE, "a") as f:
            f.write(alert_msg)
        return True
    return False


def auto_retrain(confidence):
    """
    Build  retraining trigger for model monitoring.
    """
    if confidence < 0.40:
        retrain_msg = f"[{datetime.utcnow()}] AUTO-RETRAIN triggered at confidence {confidence:.3f}\n"
        print(retrain_msg)
        # Log to file
        with open(ALERTS_FILE, "a") as f:
            f.write(retrain_msg)
        return {"status": "Retraining started"}


# Build  Routes


@app.route("/", methods=["GET"])
def home():
    return "API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Tokenize
    tokens = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(tokens)
    logits = outputs.logits

    # Prediction
    pred_label = int(tf.argmax(logits, axis=1)[0])
    probs = tf.nn.softmax(logits, axis=1)[0]
    confidence = float(probs[pred_label])

    #  Monitoring
    log_model_metrics(pred_label, confidence, text)
    check_alert(confidence)
    auto_retrain(confidence)

    return jsonify({"prediction": pred_label, "confidence": confidence})


#  Retraining trigger 
@app.route("/trigger_retrain", methods=["POST"])
def retrain_route():
    print(" Manual retraining triggered.")
    return {"status": "Retraining started"}


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """View monitoring metrics"""

    log_file = os.path.join(MONITOR_DIR, "metrics.csv")

    if not os.path.exists(log_file):
        return jsonify({"total_predictions": 0})

    df = pd.read_csv(log_file)
    return jsonify(
        {
            "total_predictions": len(df),
            "avg_confidence": float(df["confidence"].mean()),
            "min_confidence": float(df["confidence"].min()),
            "low_confidence_count": int((df["confidence"] < 0.50).sum()),
        }
    )


@app.route("/demo", methods=["GET"])
def demo():
    """Simple web demo interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Legal Text Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 10px;
            border: 2px solid #667eea;
            border-radius: 5px;
            font-size: 14px;
            margin: 10px 0;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background: #5568d3;
        }
        #result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        .success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .confidence {
            font-size: 18px;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Legal Text Classification</h1>
        <p style="text-align: center; color: #666;">Enter legal text below to classify</p>
        <textarea id="textInput" placeholder="Type or paste legal text here...
Example: The tenant must pay rent by the 5th of each month."></textarea>
        <button onclick="classify()"> Classify Text</button>
        <div class="loading" id="loading">
            <p> Analyzing...</p>
        </div>
        <div id="result"></div>
    </div>

    <script>
        async function classify() {
            const text = document.getElementById('textInput').value;
            if (!text.trim()) {
                alert('Please enter some text!');
                return;
            }

            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text})
                });

                const data = await response.json();
                // Hide loading
                document.getElementById('loading').style.display = 'none';

                // Show result
                const resultDiv = document.getElementById('result');
                const confidencePercent = (data.confidence * 100).toFixed(1);
                const alertClass = data.confidence < 0.6 ? 'warning' : 'success';
                resultDiv.className = alertClass;
                resultDiv.innerHTML = `
                    <div class="prediction"> Prediction: ${data.prediction}</div>
                    <div class="confidence">Confidence: ${confidencePercent}%</div>
                    ${data.alert ? '<p style="color: #856404; margin-top: 10px;"> Low confidence alert triggered!</p>' : ''}
                    ${data.retrain_triggered ? '<p style="color: #721c24; margin-top: 10px;"> Auto-retrain triggered!</p>' : ''}
                `;
                resultDiv.style.display = 'block';

            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error.message);
            }
        }

        // Allow Enter to submit (Ctrl+Enter for new line)
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
                e.preventDefault();
                classify();
            }
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
