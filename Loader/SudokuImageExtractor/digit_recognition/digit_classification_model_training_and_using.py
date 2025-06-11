"""
Use the mnist database to create a model for digit detection

Known issues:
If problems of SSL verification during mnist retrieval occur run the following command:
For MacOS:
* Replace the X with your python version
/Applications/Python\ 3.X/Install\ Certificates.command
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import os

def add_noise(images, noise_level=0.04):
    """
    Adds Gaussian noise to normalized images in [0, 1].

    - noise_level: standard deviation of the noise
    """
    noisy = images + np.random.normal(loc=0.0, scale=noise_level, size=images.shape)
    noisy = np.clip(noisy, 0.0, 1.0)  # Keep pixel values in [0, 1]
    return noisy

def add_no_digit_images(x_train, x_test, y_train, y_test):
    num_neg_train = int(x_train.shape[0] / 11)
    num_neg_test = int(x_test.shape[0] / 11)

    def generate_noise_images(num, noise_level = 0):
        """
            Generate 'no digit' images with random base intensity per image and local noise.

            Each image has a base intensity randomly chosen from 0–255,
            with Gaussian noise added around that base.

            - num: number of images to generate
            - noise_level: standard deviation of pixel variation from base (max ~127)
            """
        images = []
        for _ in range(num):
            base = np.random.randint(0, 256)  # Random base intensity for this image
            img = np.random.normal(loc = base, scale = noise_level, size = (28, 28))
            img = np.clip(img, 0, 255)  # Ensure pixel values stay valid
            images.append(img)

        images = np.array(images).astype('float32') / 255.0  # Normalize to [0, 1]
        images = images.reshape(num, 28, 28, 1)
        return images

    x_no_digit_train = generate_noise_images(num_neg_train)
    x_no_digit_test = generate_noise_images(num_neg_test)

    y_no_digit_train = np.full((num_neg_train,), 10)  # class 10 = no digit
    y_no_digit_test = np.full((num_neg_test,), 10)

    # --- Combine ---
    x_train = np.concatenate([x_train, x_no_digit_train], axis=0)
    y_train = np.concatenate([y_train, y_no_digit_train], axis=0)

    x_test = np.concatenate([x_test, x_no_digit_test], axis=0)
    y_test = np.concatenate([y_test, y_no_digit_test], axis=0)

    return x_train, y_train, x_test, y_test

def build_model():
    # === 1. Load and preprocess MNIST data ===
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    x_train, y_train, x_test, y_test = add_no_digit_images(x_train, x_test, y_train, y_test)
    x_train, x_test = add_noise(x_train), add_noise(x_test)

    y_train = to_categorical(y_train, 11) # Why 11? 10 digits + no digit classification
    y_test = to_categorical(y_test, 11)



    # === 2. Build the CNN model ===
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(11, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # === 3. Train the model ===
    print("Training the model...")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    # === 4. Save the model ===
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "digit_recognition_model/model.h5")
    model.save(file_path)
    print("Model saved in {}".format(file_path))

def predict_digit_from_image(image):
    """
    Return 10 if no digit.
    Return format: digit, probability
    """
    if not hasattr(predict_digit_from_image, "model"):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "digit_recognition_model/model.h5")
        predict_digit_from_image.model = load_model(file_path)

    model = predict_digit_from_image.model

    # Convert PIL image to NumPy array
    image_np = np.array(image)

    # Convert RGB (PIL) to BGR (OpenCV)
    # image = cv2.cvtColor(image_np, cv2.IMREAD_GRAYSCALE)
    image = image_np

    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    # Resize to 28x28
    img_resized = cv2.resize(image, (28, 28))

    # plt.figure()
    # plt.imshow(image)
    # plt.show()

    # Invert colors if background is white (like scanned paper)
    img_resized = 255 - img_resized

    #plt.figure()
    #plt.imshow(img_resized)
    #plt.show()

    # Normalize and reshape
    img_input = img_resized / 255.0
    img_input = img_input.reshape(1, 28, 28, 1)

    # === 6. Load model and predict ===
    prediction = model.predict(img_input)
    digit = np.argmax(prediction)

    if digit == 10:
        digit = 0

    # === 7. Display result ===
    #print(f"Predicted digit: {digit}")
    #plt.imshow(img_resized, cmap='gray')
    #plt.title(f"Prediction: {digit}")
    #plt.axis('off')
    #plt.show()

    #if prediction[0][digit] <= 0.5:
    #    plt.figure()
    #    plt.imshow(img_resized)
    #    plt.show()
    #    raise ValueError("Cannot determine digit.")

    return digit, prediction[0][digit]

def predict_digits(digit_images):
    #digit_images = [digit_images[0]]
    arr = np.zeros((9, 9), dtype=int)

    for idx, image in enumerate(digit_images):
        digit, _ = predict_digit_from_image(image)
        if digit <= 9:
            arr[int(idx / 9), int(idx % 9)] = int(digit)
        else:
            arr[int(idx / 9), int(idx % 9)] = 0

    #print("Predicted soduko: \n", arr)

    return arr


if __name__ == "__main__":
    build_model()