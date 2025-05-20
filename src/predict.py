import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys

def preprocess_image(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 if needed
    img = cv2.resize(img, (28, 28))

    # Invert image if background is white and digit is dark
    img = 255 - img

    # Normalize and reshape
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    return img

def predict(image_path, model_path="models/symbol_cnn_mnist.h5"):
    model = load_model(model_path)
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)

    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py test_images/sample_digit.png")
    else:
        predict(sys.argv[1])
