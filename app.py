from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Serve images folder
@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

# Load CNN model
model = load_model("cnn_model.h5")
print("Model loaded successfully")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if "ecg" not in request.files:
        return jsonify({'prediction': 'No file received'}), 400

    file = request.files['ecg']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Preprocess image
        img = image.load_img(file_path, target_size=(128, 128))  # match CNN input
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # CNN prediction
        pred_prob = model.predict(img_array)[0][0]

        # Simple threshold: 0.5
        label = "normal" if pred_prob >= 0.5 else "abnormal"

        return jsonify({'prediction': label})

    except Exception as e:
        print("Error processing image:", e)
        return jsonify({'prediction': 'error'}), 500

if __name__ == "__main__":
    app.run(debug=True)