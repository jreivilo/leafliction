from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import altair as alt
import math
import os
import numpy as np
import cv2
import argparse

def convert_to_grayscale(original_image):
    gray_image = pcv.rgb2gray(original_image)
    return gray_image

def thresholding_for_leaf_shape(gray_image):
    # Apply binary thresholding
    binary_image = pcv.threshold.binary(gray_image, 120, 'light')
    #invert the binary image
    binary_image = pcv.invert(binary_image)
    return binary_image

def extract_leaf_shape(original_image, binary_image):
    # Apply the mask to isolate the leaf
    leaf_shape = pcv.apply_mask(original_image, binary_image, 'white')
    return leaf_shape

def add_leaf_image_green_to_original_image(original_image, leaf_image):
    #leaf image to binary
    leaf_image = pcv.rgb2gray(leaf_image)
    leaf_image = pcv.threshold.binary(leaf_image, 120, 'light')
    
    leaf_image = pcv.apply_mask(original_image, leaf_image, 'white')
    #white pixels to green
    leaf_image[np.where((leaf_image==[255,255,255]).all(axis=2))] = [0,255,0]
    return leaf_image

def analyse_leaf_shape(original_image, mask):
    # Analyse the leaf shape
    analysis_image = pcv.analyze.size(img=original_image, labeled_mask=mask, n_labels=1)
    return analysis_image

def acute_marking(original_image, mask):
    # Mark the acute angles
    print(f"Shape of original image: {original_image.shape}")
    print(f"Shape of mask: {mask.shape}")
    acute_image = pcv.homology.acute(img=original_image, mask=mask, win=5, threshold=180)
    return acute_image

def historgam_pixel_intensity(original_image, mask):
    # Histogram of pixel intensity
    pcv.params.debug = "plot"
    hist_figure, hist_data = pcv.visualize.histogram(img=original_image, hist_data=True)
    return hist_figure, hist_data

# Save transformations
def save_transformation_results(base_output_path, transformations):
    os.makedirs(base_output_path, exist_ok=True)
    for name, image in transformations.items():
        save_path = os.path.join(base_output_path, f"{name}.png")
        
        if isinstance(image, plt.Figure):  # For matplotlib figures
            image.savefig(save_path)
        elif isinstance(image, np.ndarray):  # For OpenCV images (numpy array)
            cv2.imwrite(save_path, image)
        elif isinstance(image, alt.Chart):  # For Altair charts
            image.save(save_path)  # Save as PNG or HTML based on `save_path` extension
        else:
            print(f"Skipping unsupported format for {name}")

# Apply transformations and save results
def apply_transformations(image_path, base_output_dir="plot/transformation"):
    pcv.params.debug = "none"
   
    print(f"Processing image: {os.path.basename(image_path)}")

    original_image = cv2.imread(image_path)
    
    # Extract the type of image and image name
    image_type = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Define output directory based on image type and name
    output_dir = os.path.join(base_output_dir, image_type, image_name)
    os.makedirs(output_dir, exist_ok=True)

    # Perform transformations
    gray_image = convert_to_grayscale(original_image)
    binary_image = thresholding_for_leaf_shape(gray_image)
    leaf_image = extract_leaf_shape(original_image, binary_image)
    leaf_shape_green_background = add_leaf_image_green_to_original_image(original_image, leaf_image)
    leaf_shape_analysis = analyse_leaf_shape(original_image, binary_image)
    edges = pcv.canny_edge_detect(original_image)
    hist_figure, _ = historgam_pixel_intensity(original_image, binary_image)

    # Organize transformations into a dictionary
    transformations = {
        "gray_image": gray_image,
        "binary_image": binary_image,
        "leaf_image": leaf_image,
        "leaf_shape_green_background": leaf_shape_green_background,
        "leaf_shape_analysis": leaf_shape_analysis,
        "edges": edges,
        "histogram": hist_figure
    }

    # Save transformations to the specified output directory
    save_transformation_results(output_dir, transformations)

# Main function for handling input
def main(input_path, base_output_dir="plot/transformation"):
    if os.path.isfile(input_path):
        apply_transformations(input_path, base_output_dir)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    apply_transformations(image_path, base_output_dir)
    else:
        print(f"The path {input_path} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply various augmentations to an image or directory of images.")
    parser.add_argument("input_path", type=str, help="Path to the image file or directory of images to be transformed.")
    parser.add_argument("-o", "--output_dir", type=str, help="Directory to save transformed images if input is a directory.")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else "plot/transformation"
    main(args.input_path, output_dir)
