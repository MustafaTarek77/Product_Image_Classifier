# Product_Image_Classifier
This repository contains image classification model using a Support Vector Machine (SVM) model using Histogram of Oriented Gradients (HOG) features for image classification, and for making predictions using the trained model.

Importing Libraries: The code begins by importing necessary libraries, including scikit-learn's LinearSVC for SVM, hog from skimage.feature for extracting HOG features, joblib for model serialization, os for file operations, and cv2 for image processing.

**train_images() Function:**
Data Preparation: This function reads images from a specified directory (./Images), where each subdirectory represents a different class or label. It iterates through all images, converting them to grayscale and resizing them to a fixed size of 28x28 pixels.
Feature Extraction: HOG features are computed for each image using the hog() function. HOG is a feature descriptor that captures the shape and gradient information within local regions of an image.
Data Splitting: The dataset is split into training and validation sets using train_test_split() from scikit-learn. By default, 20% of the data is used for validation.
Model Training with Grid Search Cross-Validation: Grid search with cross-validation is performed to find the optimal hyperparameters for the Linear SVM model. The parameter grid includes different values for the regularization parameter C.
Best Model Selection: The best model from the grid search is identified based on the highest mean cross-validated score.
Model Evaluation: The accuracy of the best model is evaluated on the validation set.
Model Serialization: The best model is serialized using joblib.dump() and saved as "HOG_Model_PRODUCTS_best.npy".

**get_predict() Function:**
Loading the Model: The serialized model is loaded from the file "HOG_Model_PRODUCTS.npy" using joblib.load().
Preprocessing the Input Image: The input image is resized to 28x28 pixels to match the training data.
Feature Extraction: HOG features are extracted from the resized image.
Prediction: The trained SVM model makes predictions on the HOG features of the input image using predict(). The predicted class label is returned.

Explanation of HOG (Histogram of Oriented Gradients):
HOG is a widely used feature descriptor for object detection and image classification tasks.
It works by dividing the image into small cells, computing gradient orientations within each cell, and then constructing a histogram of gradient orientations.
These histograms are normalized and concatenated to form the final feature vector, which captures the local shape and gradient information in the image.
