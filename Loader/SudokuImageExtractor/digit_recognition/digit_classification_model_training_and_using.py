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
from PIL import Image, ImageDraw, ImageFont
import glob

def add_noise(images, noise_level=0.04):
    """
    Adds Gaussian noise to normalized images in [0, 1].

    Args:
        images (numpy.ndarray): Input images with shape (batch_size, height, width, channels)
        noise_level (float): Standard deviation of the Gaussian noise (default: 0.04)

    Returns:
        numpy.ndarray: Images with added noise, clipped to [0, 1] range
    """
    noisy = images + np.random.normal(loc=0.0, scale=noise_level, size=images.shape)
    noisy = np.clip(noisy, 0.0, 1.0)  # Keep pixel values in [0, 1]
    return noisy

def add_no_digit_images(x_train, x_test, y_train, y_test):
    """
    Adds synthetic "no digit" images to the training and test datasets.
    
    This function generates random noise images to represent empty cells in Sudoku grids,
    helping the model learn to distinguish between digits and empty spaces.

    Args:
        x_train (numpy.ndarray): Training images
        x_test (numpy.ndarray): Test images  
        y_train (numpy.ndarray): Training labels
        y_test (numpy.ndarray): Test labels

    Returns:
        tuple: (x_train_combined, y_train_combined, x_test_combined, y_test_combined)
               Combined datasets with "no digit" class (label 10)
    """
    num_neg_train = int(x_train.shape[0] / 11)
    num_neg_test = int(x_test.shape[0] / 11)

    def generate_noise_images(num, noise_level = 0):
        """
        Generate 'no digit' images with random base intensity per image and local noise.

        Each image has a base intensity randomly chosen from 0–255,
        with Gaussian noise added around that base.

        Args:
            num (int): Number of images to generate
            noise_level (int): Standard deviation of pixel variation from base (max ~127)

        Returns:
            numpy.ndarray: Generated noise images with shape (num, 28, 28, 1)
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

def build_model_using_sudoku_examples():
    x = []
    y = []

    digit_dataset_dir = os.path.join(os.path.dirname(__file__), "../../Datasets/Dataset1-SudokuImageDataset-Kaggle/digit_dataset")
    for file_name in os.listdir(digit_dataset_dir):
        if file_name.endswith(".jpg"):
            img = Image.open(os.path.join(digit_dataset_dir, file_name)).convert('L')  # Convert to grayscale like MNIST
            img = Image.eval(img, lambda x: 255 - x)  # Invert black to white and vice versa
            data_file_name = file_name.replace(".jpg", ".dat")
            digit = 0
            with open(os.path.join(digit_dataset_dir, data_file_name), 'r') as f:
                digit = int(f.read())
            # Resize to 28x28 and convert to numpy array
            img_resized = img.resize((28, 28))
            x.append(np.array(img_resized))
            y.append(digit)

    # Divide to train and test
    x_train = x[:int(len(x) * 0.8)]
    y_train = y[:int(len(y) * 0.8)]
    x_test = x[int(len(x) * 0.8):]
    y_test = y[int(len(y) * 0.8):]

    # Convert to numpy arrays and reshape
    x_train = np.array(x_train).reshape(-1, 28, 28, 1) / 255.0
    x_test = np.array(x_test).reshape(-1, 28, 28, 1) / 255.0
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Print an example of x_train where y_train is not 10
    # plt.figure()
    # idx = np.where(y_train != 10)[0][100]
    # example_image = x_train[idx]
    # plt.imshow(example_image, cmap='gray')
    # plt.show()

    x_train, y_train, x_test, y_test = add_no_digit_images(x_train, x_test, y_train, y_test)
    x_train, x_test = add_noise(x_train), add_noise(x_test)

    y_train = to_categorical(y_train, 11) # Why 11? 10 digits + no digit classification
    y_test = to_categorical(y_test, 11)



    # === 2. Build the CNN model ===
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(11, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # === 3. Train the model ===
    print("Training the model...")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=15)

    # === 4. Save the model ===
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "digit_recognition_model/model.keras")
    model.save(file_path)
    print("Model saved in {}".format(file_path))


def build_model():
    """
    Builds, trains, and saves a CNN model for digit recognition.
    
    This function:
    1. Loads and preprocesses MNIST data
    2. Adds synthetic "no digit" images and noise
    3. Builds a CNN architecture with Conv2D, MaxPooling2D, and Dense layers
    4. Trains the model for 5 epochs
    5. Saves the trained model to disk
    
    The model is designed to classify digits 0-9 plus an additional class for "no digit",
    making it suitable for Sudoku grid digit recognition.
    
    Returns:
        None: Model is saved to 'digit_recognition_model/model.keras'
    """
    # === 1. Load and preprocess MNIST data ===
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

    # Show a random image from x_train
    plt.figure()
    plt.imshow(x_train[np.random.randint(0, len(x_train))], cmap='gray')
    plt.show()

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
    file_path = os.path.join(base_dir, "digit_recognition_model/model.keras")
    model.save(file_path)
    print("Model saved in {}".format(file_path))

def build_model2():
    """
    Builds, trains, and saves an advanced CNN model for digit recognition with improved performance.
    
    This function uses:
    1. Multiple datasets (MNIST, EMNIST, custom synthetic data)
    2. Advanced data augmentation (rotation, scaling, noise, blur)
    3. Complex CNN architecture with residual connections
    4. Advanced training techniques (learning rate scheduling, early stopping)
    5. Longer training with better regularization
    
    The model is designed to achieve superior accuracy for Sudoku digit recognition
    across various image qualities and styles.
    
    Returns:
        None: Model is saved to 'digit_recognition_model/model2.keras'
    """
    from tensorflow.keras.layers import BatchNormalization, Dropout, Add, Input, GlobalAveragePooling2D
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    # === 1. Load multiple datasets ===
    print("Loading MNIST dataset...")
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    
    # Try to load EMNIST for additional data
    try:
        print("Loading EMNIST dataset...")
        from tensorflow.keras.datasets import emnist
        (x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist) = emnist.load_data(type='digits')
        
        # Combine datasets
        x_train = np.concatenate([x_train_mnist, x_train_emnist], axis=0)
        y_train = np.concatenate([y_train_mnist, y_train_emnist], axis=0)
        x_test = np.concatenate([x_test_mnist, x_test_emnist], axis=0)
        y_test = np.concatenate([y_test_mnist, y_test_emnist], axis=0)
        print(f"Combined dataset size: {len(x_train)} training, {len(x_test)} test samples")
    except:
        print("EMNIST not available, using MNIST only")
        x_train, y_train = x_train_mnist, y_train_mnist
        x_test, y_test = x_test_mnist, y_test_mnist
    
    # === 2. Preprocess data ===
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Add "no digit" class
    x_train, y_train, x_test, y_test = add_no_digit_images(x_train, x_test, y_train, y_test)
    
    # === 3. Advanced data augmentation ===
    def create_augmented_data(images, labels, augmentation_factor=3):
        """Create augmented versions of the training data"""
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest',
            brightness_range=[0.8, 1.2],
            preprocessing_function=lambda x: add_noise(x, noise_level=0.02)
        )
        
        augmented_images = []
        augmented_labels = []
        
        for i in range(len(images)):
            # Add original image
            augmented_images.append(images[i])
            augmented_labels.append(labels[i])
            
            # Generate augmented versions
            for j in range(augmentation_factor):
                aug_img = datagen.random_transform(images[i])
                augmented_images.append(aug_img)
                augmented_labels.append(labels[i])
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    print("Creating augmented training data...")
    x_train_aug, y_train_aug = create_augmented_data(x_train, y_train, augmentation_factor=2)
    
    # Convert to categorical
    y_train_aug = to_categorical(y_train_aug, 11)
    y_test = to_categorical(y_test, 11)
    
    # === 4. Build advanced CNN architecture ===
    def residual_block(x, filters, kernel_size=3):
        """Residual block with batch normalization and dropout"""
        shortcut = x
        
        x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Add shortcut connection if dimensions match
        if shortcut.shape[-1] == filters:
            x = Add()([shortcut, x])
        
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Initial convolution
    x = Conv2D(64, 7, strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = MaxPooling2D(2, 2)(x)
    
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = MaxPooling2D(2, 2)(x)
    
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    # Global average pooling and dense layers
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(11, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # === 5. Compile with advanced optimizer ===
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # === 6. Callbacks for better training ===
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # === 7. Train the model ===
    print("Training advanced model...")
    print(f"Training on {len(x_train_aug)} samples")
    print(f"Model parameters: {model.count_params():,}")
    
    history = model.fit(
        x_train_aug, y_train_aug,
        validation_data=(x_test, y_test),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # === 8. Evaluate and save ===
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # Save the model
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "digit_recognition_model/model2.keras")
    model.save(file_path)
    print(f"Advanced model saved in {file_path}")
    
    return model, history

def predict_digit_from_image(image):
    """
    Predicts a digit from a single image using the trained CNN model.
    
    This function preprocesses the input image (resize, invert, normalize) and
    uses the trained model to predict the digit. Returns 0 for "no digit" cases.
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image to classify
        
    Returns:
        tuple: (digit, probability) where:
            - digit (int): Predicted digit (0-9, where 0 can mean "no digit")
            - probability (float): Confidence score for the prediction
            
    Note:
        The model is loaded lazily on first call and cached for subsequent calls.
    """
    if not hasattr(predict_digit_from_image, "model"):
        base_dir = os.path.dirname(__file__)
        file_path = os.path.join(base_dir, "digit_recognition_model/model.keras")
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

    # Show the image as given to the model
    # plt.figure()
    # plt.imshow(img_input[0, :, :, 0], cmap='gray')  # Remove batch dimension and show as grayscale
    # plt.title('Input to Model')
    # plt.axis('off')
    # plt.show()

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
    """
    Predicts digits for a list of images representing a Sudoku grid.
    
    This function processes a list of 81 digit images (9x9 grid) and returns
    a 2D numpy array representing the predicted Sudoku board.
    
    Args:
        digit_images (list): List of 81 images, each representing a cell in the Sudoku grid
        
    Returns:
        numpy.ndarray: 9x9 array of integers representing the predicted Sudoku board
                      where 0 indicates an empty cell
        
    Note:
        The function assumes the images are in row-major order (left to right, top to bottom).
    """
    #digit_images = [digit_images[1]]
    arr = np.zeros((9, 9), dtype=int)

    for idx, image in enumerate(digit_images):
        digit, _ = predict_digit_from_image(image)
        if digit <= 9:
            arr[int(idx / 9), int(idx % 9)] = int(digit)
        else:
            arr[int(idx / 9), int(idx % 9)] = 0

    #print("Predicted soduko: \n", arr)

    return arr

def build_model3():
    """
    Builds, trains, and saves a CNN model for digit recognition using a synthetic dataset of typed (font-rendered) digits.
    The model is trained to classify digits 0-9 and a 'no digit' class (10).
    The dataset is generated by rendering digits with various fonts and random transformations.
    """
    # 1. Generate synthetic dataset of typed digits
    img_size = 28
    num_classes = 11  # 0-9 + no digit
    samples_per_digit = 5000
    fonts = glob.glob("/Library/Fonts/*.ttf")  # Adjust path for your OS if needed

    x_list = []
    y_list = []

    for digit in range(10):
        for _ in range(samples_per_digit):
            img = Image.new('L', (img_size, img_size), color=0)  # Dark background
            draw = ImageDraw.Draw(img)
            font_path = np.random.choice(fonts)
            try:
                font = ImageFont.truetype(font_path, np.random.randint(22, 32))
            except Exception:
                continue  # skip problematic fonts
            # Get the bounding box of the text to calculate its dimensions
            bbox = draw.textbbox((0, 0), str(digit), font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            
            # Calculate center position with some random offset for better training
            # Offset by bbox origin to get proper positioning
            center_x = (img_size - w) // 2 - bbox[0]
            center_y = (img_size - h) // 2 - bbox[1] - 1
            
            # Add small random offset to simulate real-world positioning variations
            offset_x = np.random.randint(-2, 3)  # -2 to +2 pixels
            offset_y = np.random.randint(-2, 3)  # -2 to +2 pixels
            
            # Ensure the digit stays within image bounds
            # pos_x = max(2, min(img_size - w - 2, center_x + offset_x))
            # pos_y = max(2, min(img_size - h - 2, center_y + offset_y))
            pos_x = center_x + offset_x
            pos_y = center_y + offset_y
            
            pos = (pos_x, pos_y)
            draw.text(pos, str(digit), fill=255, font=font)  # Bright digit
            # Reduced random rotation for better centering
            img = img.rotate(np.random.uniform(-5, 5), fillcolor=0, center=(img_size//2, img_size//2))  # Dark fill, rotate around center

            # Add gaussian noise
            img_array = np.array(img).astype('float32') / 255.0
            img_array = add_noise(img_array.reshape(1, img_size, img_size, 1)).reshape(img_size, img_size)
            img_array = np.clip(img_array * 255, 0, 255).astype('uint8')
            img = Image.fromarray(img_array)

            # Convert to array
            arr = np.array(img).astype('float32') / 255.0
            arr = arr.reshape(img_size, img_size, 1)

            # Show the image as given to the model, only for digit 2 and first iteration
            """if digit == 2 and _ == 0:
                plt.figure()
                plt.imshow(arr[:, :, 0], cmap='gray')  # Show as grayscale
                plt.title('Input to Model')
                plt.axis('off')
                plt.show()"""

            x_list.append(arr)
            y_list.append(digit)

    # Add 'no digit' class with diverse samples
    for _ in range(int(samples_per_digit / 10)):
        # Create different types of "no digit" images
        img_type = np.random.choice(['blank', 'noise', 'dots'])
        
        if img_type == 'blank':
            # Pure black or dark gray background
            color = np.random.randint(0, 16)
            img = Image.new('L', (img_size, img_size), color=color)
            
        elif img_type == 'noise':
            # Noisy background with random pixels
            img = Image.new('L', (img_size, img_size), color=0)
            pixels = np.random.randint(0, 256, (img_size, img_size))
            # Add some random bright pixels to simulate noise
            noise_mask = np.random.random((img_size, img_size)) < 0.1
            num_noise_pixels = int(np.sum(noise_mask))
            if num_noise_pixels > 0:
                pixels[noise_mask] = np.random.randint(156, 256, num_noise_pixels)
            img = Image.fromarray(pixels.astype('uint8'))
                
        elif img_type == 'dots':
            # Scattered dots or small marks
            img = Image.new('L', (img_size, img_size), color=0)
            draw = ImageDraw.Draw(img)
            num_dots = np.random.randint(3, 8)
            for _ in range(num_dots):
                x, y = np.random.randint(2, img_size-2), np.random.randint(2, img_size-2)
                radius = np.random.randint(1, 3)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=255)
        
        # Add some random rotation and slight transformations
        if np.random.random() < 0.3:
            img = img.rotate(np.random.uniform(-10, 10), fillcolor=255)

        # Add some random noise
        img_array = np.array(img).astype('float32') / 255.0
        img_array = add_noise(img_array.reshape(1, img_size, img_size, 1)).reshape(img_size, img_size)
        img_array = np.clip(img_array * 255, 0, 255).astype('uint8')
        img = Image.fromarray(img_array)
        
        # Convert to array
        arr = np.array(img).astype('float32') / 255.0
        arr = arr.reshape(img_size, img_size, 1)

        # Show the image as given to the model, only for first iteration
        """if _ == 0:
            plt.figure()
            plt.imshow(arr[:, :, 0], cmap='gray')  # Show as grayscale
            plt.title('Input to Model')
            plt.axis('off')
            plt.show()"""

        x_list.append(arr)
        y_list.append(10)

    x = np.array(x_list)
    y = np.array(y_list)

    # Shuffle and split
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    split = int(0.9 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    # Print a random image from x_train
    # plt.figure()
    # plt.imshow(x_train[np.random.randint(0, len(x_train))], cmap='gray')
    # plt.show()

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # 2. Build the CNN model (same as build_model)
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train the model
    print("Training the model on typed digits...")
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)

    # 4. Save the model
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "digit_recognition_model/model3.keras")
    model.save(file_path)
    print("Model saved in {}".format(file_path))

if __name__ == "__main__":
    import sys
    # build_model()
    build_model_using_sudoku_examples()
    
    """if len(sys.argv) > 1 and sys.argv[1] == "advanced":
        print("Training advanced model...")
        build_model2()
    else:
        print("Training basic model...")
        build_model()"""