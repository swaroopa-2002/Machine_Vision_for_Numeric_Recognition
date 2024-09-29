# Machine Vision for Numeric Recognition

This project focuses on implementing machine learning models to recognize numeric digits from real-world images, specifically from the **SVHN dataset**. By comparing the performance of **Multilayer Perceptron (MLP)** and **Convolutional Neural Network (CNN)** models, we aim to determine the most effective approach for accurate digit recognition in complex scenes, such as those found in street view images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Introduction
Accurate digit recognition plays a crucial role in applications such as automated mapping and navigation systems. This project evaluates MLP and CNN architectures using the **SVHN dataset** to identify which model is more effective for real-world numeric digit recognition in various scenes.

## Dataset
We used the **Street View House Numbers (SVHN)** dataset:

- **Link to Dataset**: [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
- **Classes**: 10 classes representing digits from 0 to 9.
- **Dataset Size**: 
  - **Training Set**: 73,257 images
  - **Test Set**: 26,032 images
  - **Extra Set**: 531,131 images

The dataset comes in two formats:

1. **Format 1**: Original images with character-level bounding boxes.
2. **Format 2**: MNIST-like 32x32 images centered around a single digit.

## Data Preprocessing
We implemented several data preprocessing steps to ensure the models perform optimally:

- **Grayscale Conversion**: Convert RGB images to grayscale for simplicity and efficiency.
- **Normalization**: Scale pixel values to the [0, 1] range.
- **Label Encoding**: Convert labels into one-hot encoded vectors.
- **Image Rearrangement**: Ensure the correct input shape for model compatibility.

## Methodology
We implemented two different models to recognize the digits:

### **MLP (Multilayer Perceptron)**
- Fully connected layers for classification.
- Trained for 30 epochs.
- Early stopping and dynamic learning rate reduction.

### **CNN (Convolutional Neural Network)**
- Convolutional layers to capture spatial features.
- Trained for 10 epochs.
- Early stopping and model checkpoints to optimize performance.

Both models were compiled using:
- **Optimizer**: Adam
- **Learning rate**: 0.002
- **Loss function**: Categorical Crossentropy

## Results
- **MLP**: Lower accuracy in recognizing digits, especially in classes with fewer examples.
- **CNN**: Consistently higher accuracy across all classes due to its ability to process spatial data effectively.

Performance was evaluated using:
- **Accuracy**: Overall model accuracy on the test set.
- **Confusion Matrix**: Visual representation of correct/incorrect predictions.
- **Classification Report**: Detailed accuracy, recall, and F1-score for each class.

## Conclusion
The **CNN** outperformed the **MLP** in recognizing digits from the **SVHN dataset**, particularly in more complex and varied scenes. Its convolutional layers allowed it to capture spatial relationships in the images, making it the better choice for real-world image recognition tasks.

Future work could explore more complex CNN architectures or incorporate other machine learning algorithms for further improvements.

## Installation
To run the project locally, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository-name.git
