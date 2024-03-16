from sklearn.svm import LinearSVC
from skimage.feature import hog
import joblib
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def train_images():
    images = []
    labels = []

    # get all the image folder paths
    image_paths = os.listdir("./Images")

    for path in image_paths:
        # get all the image names
        all_images = os.listdir(f"./Images/{path}")

        # iterate over the image names, get the label
        for image in all_images:
            image_path = f"./Images/{path}/{image}"
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))

            # get the HOG descriptor for the image
            hog_desc = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')

            # update the data and labels
            images.append(hog_desc)
            labels.append(path)

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the parameter grid to search
    param_grid = {'C': [0.1, 1, 10, 100]}

    # Initialize the grid search cross-validation
    grid_search = GridSearchCV(LinearSVC(random_state=42, tol=1e-5), param_grid, cv=5)

    # Perform grid search cross-validation
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Validate the model
    val_accuracy = best_model.score(X_val, y_val)
    print(f'Validation Accuracy: {val_accuracy}')

    # Save the best model
    joblib.dump(best_model, "HOG_Model_PRODUCTS_best.npy")


def get_predict(image):
    model_path = os.path.join(os.getcwd(), "HOG_Model_PRODUCTS.npy")
    HOG = joblib.load(model_path)

    resized_image = cv2.resize(image, (28, 28))
    # get the HOG descriptor for the test image
    hog_desc = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    # prediction
    pred = HOG.predict(hog_desc.reshape(1, -1))[0]

    # print(pred.title())
    return pred.title()
