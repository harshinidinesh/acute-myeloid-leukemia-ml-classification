import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy


# Parse the .sums file to extract metadata
def parse_sums_file(sums_file):
    file_info_list = []
    with open(sums_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                file_info_list.append(parts)
            else:
                print(f"Skipping invalid line in .sums file: {line}")
    return file_info_list


# Load image and label using the correct path
def load_image_and_label(file_info, base_dir, img_size=(100, 100)):
    image_path = file_info[1]
    image_path = image_path.replace("data/", "")
    full_image_path = os.path.join(base_dir, image_path)
    print(f"Loading image from path: {full_image_path}")

    # skips missing images if they exist in the .sums file but not in the directory
    if not os.path.exists(full_image_path):
        print(f"Skipping missing image: {image_path}")
        return None, None

    # preprocess the image: resize, convert to grayscale, and normalize pixel values
    try:
        img = Image.open(full_image_path)
        img = img.resize(img_size)
        img = img.convert("L")
        img_array = np.array(img) / 255.0
        category = os.path.basename(os.path.dirname(os.path.dirname(image_path)))

        label = category
        return img_array, label

    except Exception as e:
        print(f"Error loading image at {full_image_path}: {e}")
        return None, None


def main():
    sums_file = "C:/Users/dines/OneDrive/Documents/NEU/SPRING 2026 SEM/DS 4420/final_proj/AML-Cytomorphology_MLL_Helmholtz_v1.sums"
    base_dir = "C:/Users/dines/OneDrive/Documents/NEU/SPRING 2026 SEM/DS 4420/final_proj/data"

    if not os.path.exists(base_dir):
        print(f"Error: The base directory does not exist: {base_dir}")
        return


    file_info_list = parse_sums_file(sums_file)

    images = []
    labels = []

    for file_info in file_info_list[:500]:
        img, label = load_image_and_label(file_info, base_dir)
        if img is not None:
            images.append(img)
            labels.append(label)


    print(f"Successfully loaded {len(images)} images.")

    # Convert lists to numpy arrays and encode labels
    x_data = np.array(images)
    label_map = {label: idx for idx, label in enumerate(set(labels))}
    y_data = np.array([label_map[label] for label in labels])

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.25, random_state=42
    )

    # identify the number of rows and columns in the images for reshaping
    img_rows, img_cols = x_train.shape[1], x_train.shape[2]

    # reshapes data to fit the input shape of the CNN (number of samples, rows, cols, channels)
    x_train = x_train.reshape(-1, img_rows, img_cols, 1)
    x_test = x_test.reshape(-1, img_rows, img_cols, 1)

    # builds a CNN model with two convolutional layers, max pooling, and fully connected layers for classification
    inpx = Input(shape=(img_rows, img_cols, 1))
    conv_layer = Conv2D(32, (3,3), activation='relu', padding='same')(inpx)
    pool_layer = MaxPooling2D((3,3))(conv_layer)
    # added a second convolutional layer to increase the model's capacity 
    # to learn more complex features from the images, which can improve classification performance
    second_conv_layer = Conv2D(64, (3,3), activation='relu', padding='same')(pool_layer)
    second_pool_layer = MaxPooling2D((2,2))(second_conv_layer)
    # flattens to convert the 2D feature maps into a 1D vector
    flat_G = Flatten()(second_pool_layer)
    # added first hidden layer with 128 neurons and 'relu' activation to allow model complexity
    hid_layer = Dense(128, activation='relu')(flat_G)
    # added a second hidden layer with 64 neurons and 'tanh' activation to allow model complexity
    hid_layer2 = Dense(64, activation='tanh')(hid_layer)
    # output layer with softmax activation for multi-class classification, where the number of neurons corresponds to the number of unique labels
    # we do this because AML has multiple subtypes, so we need to classify into more than two categories (unlike binary), and softmax allows us to get probabilities for each class
    out_layer = Dense(len(label_map), activation='softmax')(hid_layer2)
    model = Model([inpx], out_layer)

    # compiles the model using the Adam optimizer for less cross-entropy loss and tracks accuracy as a metric 
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # fits the model to training data for 10 epochs
    # model will learn from data and adjust weights to minimize loss, with batch size of 64 for efficient training
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=64,
        verbose=1
    )

    # prints test accuracy of the model on unseen data to evaluate performance and generalization capability
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


    # visualizes the first 9 images from the test set along with their true labels 
    # to qualitatively assess the model's performance and understand the data distribution
    plt.figure(figsize=(10,10))
    
    # (i.e. we can see if the model is correctly classifying different subtypes of AML by looking at the images and their corresponding labels, 
    # which can help identify any patterns or misclassifications in the data)
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[i].reshape(img_rows,img_cols), cmap='gray')
        plt.title(f"Label: {y_test[i]}")
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()