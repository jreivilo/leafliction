import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import os
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2


def extract_leaf(original_image):
	# Convert to LAB color space
	lab_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)
	l_channel, a_channel, b_channel = cv2.split(lab_image)

	# Convert a_channel to 8-bit
	a_channel_8bit = cv2.convertScaleAbs(a_channel)

	# Thresholding on the a-channel (which should highlight the green parts)
	_, binary_image = cv2.threshold(a_channel_8bit, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Morphological operations to clean up the mask
	kernel = np.ones((5,5), np.uint8)
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

	# Find contours
	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Filter contours based on area or other criteria
	leaf_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

	# Create a mask from the contours
	mask = np.zeros_like(binary_image)
	cv2.drawContours(mask, leaf_contours, -1, (255), thickness=cv2.FILLED)

	# Apply the mask to the original image
	leaf_shape = cv2.bitwise_and(original_image, original_image, mask=mask)

	
	return leaf_shape

def load_test_data(test_dir, batch_size=1):
	"""
	Prepares the test data for evaluation.
	
	:param test_dir: Directory containing the test data.
	:param batch_size: Number of images per batch (default 20).
	:return: Returns a generator for the test data.
	"""
	test_datagen = ImageDataGenerator(rescale=1.0/255,
									preprocessing_function=extract_leaf)
	print(test_dir)
	test_generator = test_datagen.flow_from_directory(test_dir,
													class_mode='categorical', 
													target_size=(250, 250),
													shuffle=False)
	return test_generator

def predict_and_evaluate(model, test_generator):
	"""
	Makes predictions with the model and evaluates its accuracy.

	:param model: The trained model.
	:param test_generator: Generator for the test data.
	:return: The accuracy of the model on the test data.
	"""
	# Predictions
	predictions = model.predict(test_generator)
	predicted_classes = np.argmax(predictions, axis=1)

	# True labels
	true_classes = test_generator.classes
	
	print(f"True classes: {true_classes}")
	print(f"Predicted classes: {predicted_classes}")

	# Accuracy
	accuracy = accuracy_score(true_classes, predicted_classes)
	return accuracy

import sys
import cv2
import matplotlib.pyplot as plt

validation_datagen = ImageDataGenerator(rescale=1.0/255,
										preprocessing_function=extract_leaf
										)# standardizing the pixel values

if __name__ == "__main__":
	model_path = 'model.h5'
	test_dir = 'Unit_test_all'
 
	print("No argument passed, running evaluation on test data.")
	# Load the model
	model = load_model(model_path)

	# Prepare the test data
	print("Loading test data...")
	test_generator = load_test_data(test_dir)

	# Evaluate the model
	accuracy = predict_and_evaluate(model, test_generator)
	print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")

