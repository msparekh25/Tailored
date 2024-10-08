Code Report

# Report on WikiArt Dataset Inspection and Classification Model Implementation

## Introduction

This report outlines two Python scripts developed to work with the WikiArt dataset using DeepLake and PyTorch frameworks. The first script focuses on dataset inspection, providing insights into the data structure and contents. The second script involves training a ResNet-50 model for art style classification and deploying it via a Flask API for serving predictions.

---

## 1. Dataset Inspection Script: `deeplake_wikiart_dataset_inspection.py`

### Purpose

The primary goal of this script is to load the WikiArt dataset from DeepLake and provide a comprehensive overview of its structure and contents. By inspecting the dataset, users can understand the data they are working with, which is crucial before proceeding to model training.

### Methods Used

#### **a. DeepLake Library**

- **Dataset Loading**: Utilizes `deeplake.load('hub://activeloop/wiki-art')` to access the dataset hosted on Activeloop's platform.
  
#### **b. Dataset Information Retrieval**

- **Detailed Information**: Calls `ds.info(verbose=True)` to print detailed information about the dataset, including tensor shapes, data types, and sample data.
  
- **Summary Statistics**: Uses `ds.summary()['total_samples']` to obtain the total number of samples.
  
- **Field Extraction**: Retrieves available tensor fields (e.g., `'images'`, `'labels'`) using `ds.tensors.keys()`.

#### **c. Sample Retrieval and Display**

- **Random Sampling**: Employs `ds.sample(n=5, seed=42)` to randomly select five samples from the dataset for inspection.
  
- **Data Iteration**: Iterates over the sampled data and prints each field's contents, allowing users to visualize actual data entries.

### Benefits

- **Understanding Data Structure**: Provides clarity on the dataset's composition, aiding in identifying any preprocessing needs.
  
- **Data Validation**: Ensures the dataset is as expected, which helps prevent issues during model training.

---

## 2. Model Training and Deployment Script: `train_and_deploy_wikiart_classifier.py`

### Purpose

This script trains a convolutional neural network (CNN) model to classify artworks into different styles based on images from the WikiArt dataset. It also deploys the trained model using a Flask API, enabling users to make predictions on new images.

### Methods Used

#### **a. Libraries and Frameworks**

- **DeepLake**: For efficient data loading and preprocessing.

- **PyTorch and PyTorch Lightning**: For defining the model architecture, training loops, and leveraging GPU acceleration.

- **Torchvision**: Provides the pre-trained ResNet-50 model and image transformation utilities.

- **Flask**: Serves as the web framework to deploy the model for inference.

- **Scikit-learn**: Used for computing evaluation metrics like accuracy, precision, recall, and F1-score.

#### **b. Data Loading and Preprocessing**

- **Dataset Access**: Loads the dataset in read-only mode to prevent accidental modifications.

- **Transformations**:
  - **Training**: Applies resizing, random horizontal flips, color jittering, conversion to tensor, and normalization.
  - **Inference**: Uses resizing, conversion to tensor, and normalization to prepare input images for prediction.

- **Assigning Transforms**: Associates the transformations with the dataset's `'images'` tensor via `ds.transforms`.

- **Data Splitting**: Splits the dataset into training and validation sets using an 80-20 ratio with `ds.random_split([0.8, 0.2])`.

- **DataLoaders**: Creates PyTorch DataLoaders to handle batching and shuffling for both training and validation datasets.

#### **c. Model Definition**

- **Model Selection**: Uses a pre-trained ResNet-50 model from Torchvision for transfer learning.

- **Architecture Modification**: Replaces the final fully connected layer to match the number of art style classes in the dataset.

- **LightningModule**: Wraps the model within a PyTorch Lightning `LightningModule` to streamline training and validation steps.

#### **d. Training Process**

- **Optimizer and Loss Function**:
  - Uses **Adam optimizer** with a learning rate of 0.001.
  - Employs **CrossEntropyLoss** as the criterion for classification tasks.

- **Training Loop**:
  - **Training Step**: Processes each batch, computes loss, and backpropagates errors.
  - **Validation Step**: Evaluates the model on the validation set without updating weights.
  - **Metrics Logging**: Logs loss and accuracy for both training and validation phases.

- **Trainer Configuration**: Sets up the PyTorch Lightning `Trainer` with appropriate hardware accelerators and specifies the number of epochs.

#### **e. Model Evaluation**

- **Metrics Computation**: After each epoch, calculates overall accuracy, precision, recall, and F1-score using scikit-learn's metrics functions.

- **Logging Metrics**: Records evaluation metrics for monitoring model performance over time.

#### **f. Model Saving**

- **Checkpointing**: Saves the trained model's state dictionary using PyTorch Lightning's checkpointing mechanism.

#### **g. Deployment with Flask**

- **API Endpoint**: Defines a `/predict` route that accepts POST requests with image files.

- **Image Processing**:
  - Reads the uploaded image and applies the same preprocessing transformations used during training.
  - Converts the image to the appropriate tensor format and moves it to the designated device (CPU or GPU).

- **Prediction**:
  - The model predicts the art style of the input image.
  - Retrieves the corresponding class name from `class_names`.

- **Response Handling**: Returns the predicted art style in JSON format. Handles errors gracefully by providing meaningful error messages.

### Benefits

- **Efficient Training**: Leverages pre-trained models and transfer learning to reduce training time and improve performance.

- **Modular Code Structure**: Organizes code into classes and functions for better readability and maintenance.

- **Scalability**: Can easily adjust to larger datasets or different model architectures.

- **Accessibility**: The Flask API allows integration with web applications or services, making the model accessible to end-users.

---

## Conclusion

The two scripts collectively provide a full pipeline for working with the WikiArt dataset:

1. **Dataset Inspection**: Offers insights into the dataset, ensuring a clear understanding of the data before modeling.

2. **Model Training and Deployment**: Trains a robust CNN model for art style classification and deploys it as a web service for real-time predictions.

By using DeepLake for data handling and PyTorch Lightning for model training, the implementation is both efficient and scalable. The deployment via Flask makes the trained model readily available for practical applications, such as art analysis tools, educational platforms, or recommendation systems.

---

## Recommendations

- **Hyperparameter Tuning**: Experiment with different learning rates, batch sizes, and optimizers to potentially improve model performance.

- **Data Augmentation**: Incorporate additional augmentation techniques to enhance the model's ability to generalize.

- **Model Exploration**: Try other architectures like EfficientNet or Vision Transformers to compare performance.

- **Production Deployment**: Consider deploying the model using production-grade servers or containerization tools like Docker for scalability and robustness.

- **Monitoring and Logging**: Implement monitoring tools to track the model's performance in a production environment.

---
