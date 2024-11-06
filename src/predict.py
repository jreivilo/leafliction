import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import os
import shutil
import cv2
import sys

# Définition des noms de classes
class_names = ["Apple_black_rot", "Apple_healthy", "Apple_rust", "Apple_scab", 
               "Grape_black_rot", "Grape_esca", "Grape_healthy", "Grape_spot"]

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

	# Checks whether the generator contains images
	if test_generator.samples == 0:
		print("Error: No images found in the specified path. Please check the path and ensure it contains valid images or repository.")
		sys.exit(1)

	return test_generator

def predict_single_image_with_generator(model, image_path):
    # Obtain the expected class using the parent folder if possible
	expected_class_name = os.path.basename(os.path.dirname(image_path))
	expected_class_index = class_names.index(expected_class_name) if expected_class_name in class_names else None

    # Create a temporary directory for the single image
	temp_dir = "temp_single_image_dir"
	temp_subdir = os.path.join(temp_dir, "class_0")
	os.makedirs(temp_subdir, exist_ok=True)
    
    # Copy the image to the temporary directory
	shutil.copy(image_path, os.path.join(temp_subdir, os.path.basename(image_path)))
    
    # Use ImageDataGenerator to load the single image
	test_datagen = ImageDataGenerator(rescale=1.0/255, preprocessing_function=extract_leaf)
	test_generator = test_datagen.flow_from_directory(temp_dir, class_mode=None, target_size=(250, 250), batch_size=1, shuffle=False)
    
    # Predictions
	prediction = model.predict(test_generator)
	predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Get the class label
	class_label = class_names[predicted_class]
	expected_class_label = class_names[expected_class_index] if expected_class_index is not None else "Unknown"
    
    # Display the predicted and expected class
	print(f"Prediction for {image_path}:")
	print(f"Predicted Class {predicted_class} ({class_label})")
	print(f"Expected Class {expected_class_index} ({expected_class_label})")

    # Clean up the temporary directory
	shutil.rmtree(temp_dir)
	return predicted_class, expected_class_index

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

if __name__ == "__main__":
    model_path = 'model.h5'
    model = load_model(model_path)

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if os.path.isfile(input_path) and input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            # If only one image is supplied
            print(f"Single image provided: {input_path}")
            predict_single_image_with_generator(model, input_path)
        elif os.path.isdir(input_path):
            # If a directory is supplied
            print(f"Directory provided: {input_path}")
            test_generator = load_test_data(input_path)
            accuracy = predict_and_evaluate(model, test_generator)
            print(f"Accuracy on directory images: {accuracy * 100:.2f}%")
        else:
            print("Provided path is neither an image not a directory.")
    else:
        # Default behaviour if no arguments are passed
        test_dir = 'Unit_test_all'
        print("No argument passed, running evaluation on default test data.")
        test_generator = load_test_data(test_dir)
        accuracy = predict_and_evaluate(model, test_generator)
        print(f"Accuracy on Test Data: {accuracy * 100:.2f}%")