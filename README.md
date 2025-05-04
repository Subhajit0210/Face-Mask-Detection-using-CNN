# Face Mask Detection using CNN

## Table of Contents
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Data Collection](#data-collection)
- [Data Preparation](#data-preparation)
- [Data Visualization](#data-visualization)
- [Clustering Techniques](#clustering-techniques)
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




