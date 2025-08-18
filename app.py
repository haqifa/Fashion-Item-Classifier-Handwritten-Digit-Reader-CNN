import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from tensorflow.keras.datasets import fashion_mnist, mnist

app = Flask(__name__, static_url_path='/static')

# Load the trained models
FASHION_MODEL_PATH = 'fashion2_mnist_cnn.h5'
NUMBER_MODEL_PATH = 'number_mnist_cnn.h5'
fashion_model = load_model(FASHION_MODEL_PATH)
number_model = load_model(NUMBER_MODEL_PATH)

# Load test data for accuracy calculation
(_, _), (X_fashion_test, y_fashion_test) = fashion_mnist.load_data()
X_fashion_test = X_fashion_test.astype('float32') / 255.0
X_fashion_test = X_fashion_test.reshape(-1, 28, 28, 1)
fashion_acc = fashion_model.evaluate(X_fashion_test, y_fashion_test, verbose=0)[1]

(_, _), (X_number_test, y_number_test) = mnist.load_data()
X_number_test = X_number_test.astype('float32') / 255.0
X_number_test = X_number_test.reshape(-1, 28, 28, 1)
number_acc = number_model.evaluate(X_number_test, y_number_test, verbose=0)[1]

# Fashion MNIST class names
fashion_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneakers', 'Bag', 'Ankle boot']

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict_fashion', methods=['POST'])
def predict_fashion():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        img = Image.open(file.stream).convert('L').resize((28, 28))
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 28, 28, 1)
        probs = fashion_model.predict(img_arr)[0]
        class_idx = int(np.argmax(probs))
        class_name = fashion_class_names[class_idx]
        confidence = float(probs[class_idx]) * 100
        return jsonify({'prediction': class_name, 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_number', methods=['POST'])
def predict_number():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        img = Image.open(file.stream).convert('L').resize((28, 28))
        img_arr = np.array(img) / 255.0
        img_arr = img_arr.reshape(1, 28, 28, 1)
        probs = number_model.predict(img_arr)[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx]) * 100
        return jsonify({'prediction': str(class_idx), 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
