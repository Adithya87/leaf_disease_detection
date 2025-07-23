# Plant Leaf Disease Detection using Transfer Learning

This project demonstrates a deep learning approach to classify plant leaf diseases using a Convolutional Neural Network (CNN). It leverages transfer learning with the VGG16 model, pre-trained on ImageNet, to classify images into 33 different categories of plant diseases and healthy leaves.

The script is designed to run in an environment like Google Colab, where it can handle dataset downloads and setup automatically.

## Features

- **Model**: Utilizes a pre-trained VGG16 model for feature extraction.
- **Technique**: Employs transfer learning by freezing the base model and training a custom classifier head.
- **Frameworks**: Built with TensorFlow and Keras.
- **Dataset**: Uses the "New Plant Diseases Dataset" from Kaggle.
- **Classes**: Capable of classifying 33 distinct plant/disease types.

## Dataset

The model is trained on the [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) from Kaggle. This dataset contains thousands of augmented images of plant leaves.

> **Note**: The current script (`leaf_deases_(1).py`) is configured to use only a small subset of **100 images per class** for demonstration and quick training. For a production-ready model, it is highly recommended to use the entire dataset.

## Prerequisites

- Python 3.7+
- A Kaggle account and an API key (`kaggle.json`).
- The required Python packages can be found in `requirements.txt`.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set up Kaggle API Key:**
    - Download your `kaggle.json` API token from your Kaggle account page (`Account` -> `API` -> `Create New Token`).
    - Place the `kaggle.json` file in the root directory of this project. The script will automatically copy it to the required location (`~/.kaggle/`).

## How to Run

Execute the main Python script from your terminal:

```bash
python f/leaf_disease_detection/leaf_deases_(1).py
```

The script will perform the following steps:
1.  Set up the Kaggle API credentials.
2.  Download and extract the dataset from Kaggle.
3.  Load and preprocess a subset of the images.
4.  Build the transfer learning model using VGG16.
5.  Compile and train the model.
6.  After training, it will display 50 test images along with their actual and predicted labels.

## Model Architecture

-   **Base Model**: VGG16 with weights pre-trained on 'imagenet'. The layers of the base model are frozen and not trainable.
-   **Custom Head**:
    1.  `GlobalAveragePooling2D`: To reduce the spatial dimensions from the VGG16 feature maps.
    2.  `Flatten`: To prepare the vector for the final layer.
    3.  `Dense`: A fully connected layer with 33 output units (one for each class) and a `softmax` activation function for multi-class classification.

## Future Improvements

-   **Refactor Data Loading**: The current data loading process is highly repetitive. It should be refactored into a single, dynamic loop or, preferably, replaced with Keras's `image_dataset_from_directory` utility for better efficiency and scalability.
-   **Use Full Dataset**: Modify the script to train on the entire dataset to build a more robust and accurate model.
-   **Fine-Tuning**: After initial training, unfreeze some of the top layers of the VGG16 model and fine-tune them with a low learning rate to potentially improve accuracy.
-   **Data Augmentation**: While the dataset is already augmented, applying further real-time data augmentation using `ImageDataGenerator` can help improve model generalization.