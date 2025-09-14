# Fashion Item Classifier & Handwritten Digit Reader (CNN)

This repository contains a project for classifying fashion items and handwritten digits using Convolutional Neural Networks (CNNs). It provides both Python scripts and Jupyter notebooks for model training, evaluation, and an interface to interact with the models.

## Features

- **Handwritten Digit Recognition**: Classifies digits using a CNN model trained on MNIST-like data.
- **Fashion Item Classification**: Identifies fashion items (e.g., shirts, shoes) using a CNN model trained on the Fashion MNIST dataset.
- **Interactive Interface**: A web app (using Flask) and an HTML frontend for user interaction.
- **Pre-trained Models**: Includes `.h5` files for instant inference.
- **Jupyter Notebooks**: For exploratory analysis, model training, and evaluation.

## Project Structure

```
.
├── app.py                     # Main application script (web server/backend)
├── index.html                 # Web interface for user interaction
├── fashion_mnist_train_model.ipynb  # Notebook for training Fashion MNIST model
├── number.ipynb               # Notebook for digit recognition
├── fashion2_mnist_cnn.h5      # Pre-trained Fashion MNIST model
├── number_mnist_cnn.h5        # Pre-trained digit recognition model
├── fashion-mnist_test.csv     # Fashion MNIST test data
├── digits_png.zip             # Sample digit images
├── fashionitems_png.zip       # Sample fashion item images
├── hand3.png                  # Example input image
├── static/                    # Static files for the web app
```

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: `tensorflow`, `keras`, `flask`, `numpy`, `pandas`, etc.

Install dependencies with:

```bash
pip install -r requirements.txt
```
*(Create `requirements.txt` if missing by exporting your environment or listing needed packages.)*

### Running the Application

1. **Start the Web App**:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://localhost:5000` (or the URL provided in the terminal).

2. **Interact with the Models**:
   - Use the web interface to upload images and get predictions for digits or fashion items.

3. **Training the Models** (Optional):
   - Open `fashion_mnist_train_model.ipynb` or `number.ipynb` in Jupyter Notebook to retrain or experiment with the models.

## Data & Models

- **fashion-mnist_test.csv**: Test dataset for Fashion MNIST.
- **digits_png.zip, fashionitems_png.zip**: Example images for testing.
- **.h5 model files**: Pre-trained models for immediate use.

## Acknowledgements

- [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- TensorFlow, Keras, Flask, and other open-source libraries.

## License

This project is licensed under the MIT License.

---

*For questions or contributions, please open an issue or submit a pull request.*
