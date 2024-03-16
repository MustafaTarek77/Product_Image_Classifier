import cv2
import os
from ML_Model import *

# train_images()

all_images = os.listdir(f"./Test_Images")
for image in all_images:
    image_path = f"./Test_Images/{image}"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    print(get_predict(image))