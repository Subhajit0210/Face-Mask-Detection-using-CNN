# Face Mask Detection using CNN

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
- [Data Preparation](#data-preparation)
- [CNN Model](#cnn-model)
- [Results and Insights](#results-and-insights)
- [Usage](#usage)
- [Contributing](#contributing)

## Project Overview
The goal of this project is to create a model that can accurately detect whether a person in an image is wearing a face mask or not. This is achieved by training a CNN on a dataset of images of people with and without masks. The trained model can then be used to classify new images and predict whether a person is wearing a mask.

## Dependencies
The following libraries are required to run this project:
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Matplotlib
- PIL
- Scikit-learn

## Data Collection
The project uses the "Face Mask Dataset" available on Kaggle. This dataset contains images of people wearing and not wearing face masks. It is used to train and evaluate the CNN model. You can download it using the Kaggle API.
* **Source:** Kaggle Datasets
* **Link:** https://www.kaggle.com/datasets/omkargurav/face-mask-dataset
* **Content:** Images of people with and without face masks.
* **Classes:** 
    * with_mask: Images of individuals wearing face masks correctly.
    * without_mask: Images of individuals not wearing face masks.
    * mask_weared_incorrect: Images of individuals wearing masks incorrectly (e.g., covering the mouth but not the nose).
* **Size:** 853 images
* **Labeled Objects:** 4072 labeled objects belonging to the 3 classes mentioned above.
* **Image Format:** JPEG
* **Resolution:** Varies

## Data Preparation
* **Download:** The Dataset is downloaded from Kaggle using the Kaggle API.
* **Extract:** The downloaded dataset is extracted.
* **Resize:** All images are resized to 128x128 pixels for consistency.
* **Convert:** Images are converted into NumPy arrays for processing.
* **Label:** Images are labeled as 'with_mask' or 'without_mask'.
* **Split:** Data is split into training and testing sets.
* **Scale:** Pixel values are scaled to a range of 0-1.

## CNN Model
This project employs a Convolutional Neural Network (CNN) to classify images of people with and without masks. CNNs are particularly well-suited for image recognition tasks due to their ability to learn spatial hierarchies of features.

**Model Architecture:**

The CNN model consists of the following layers:
* **Convolutional Layers:** These layers extract features from the input images using learnable filters. We use multiple convolutional layers with increasing filter sizes to capture both low-level and high-level features.
* **Max Pooling Layers:** These layers downsample the feature maps, reducing their dimensionality and computational complexity. They also help to make the model more robust to variations in the input images.
* **Flatten Layer:** This layer converts the multi-dimensional feature maps into a single vector, preparing them for input to the fully connected layers.
* **Fully Connected Layers:** These layers perform the final classification based on the extracted features. We use two fully connected layers with ReLU activation functions, followed by a final output layer with a sigmoid activation function for binary classification (mask or no mask).

## Results and Insights
* **High Accuracy:** Our CNN model achieved over 92% accuracy in detecting face masks.
* **Effective for Real-world Use:** This shows its potential for applications like public health monitoring.
* **Further Improvements:** We can explore data augmentation and model architecture for even better results.

## Usage
To run the project, follow these steps:
1. Clone the repository:
```bash
git clone https://github.com/yourusername/customer-segmentation.git
```
2. Navigate to the project directory:
```bash
cd customer-segmentation
```
4. Run the Jupyter notebook:
```bash
jupyter notebook Customer_Segmentation.ipynb
```

## Contributing
Contributions are welcome! Please create a new branch for any changes and submit a pull request for review.
