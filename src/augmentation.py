from PIL import Image, ImageOps, ImageTransform, ImageFilter
import os
import argparse
from tqdm import tqdm
import shutil

def copy_and_process_images(original_folder_path, new_folder_path):
    total_images = 0
    for root, dirs, files in os.walk(original_folder_path):
        for filename in files:
            if filename.lower().endswith(".jpg"):
                total_images += 1

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for root, dirs, files in os.walk(original_folder_path):
            for filename in files:
                if filename.lower().endswith(".jpg"):
                    # Construct the original file path
                    original_file_path = os.path.join(root, filename)

                    # Create a new path under the new folder, maintaining the subdirectory structure
                    relative_path = os.path.relpath(root, original_folder_path)
                    new_root = os.path.join(new_folder_path, relative_path)
                    if not os.path.exists(new_root):
                        os.makedirs(new_root)
                    new_file_path = os.path.join(new_root, filename)

                    # Copy the image to the new location
                    shutil.copy2(original_file_path, new_file_path)

                    # Apply transformations to the image in its new location
                    apply_augmentations(new_file_path)

                    # Update progress
                    pbar.update(1)

def apply_radial_distortion(image, strength=1):
	# Radial distortion effect
	width, height = image.size
	x_center, y_center = width / 2, height / 2
	new_image = Image.new("RGB", (width, height))

	for x in range(width):
		for y in range(height):
			# Shift x and y coordinates by a factor based on distance from the center
			factor = 1 + strength * ((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * x_center ** 2)
			new_x, new_y = int(x_center + (x - x_center) * factor), int(y_center + (y - y_center) * factor)
			if 0 <= new_x < width and 0 <= new_y < height:
				new_image.putpixel((x, y), image.getpixel((new_x, new_y)))

	return new_image

def apply_augmentations(file_path: str) -> None:
    # Load the original image
    original_image = Image.open(file_path)
    base, extension = os.path.splitext(file_path)

    # Get the original and parent directories
    original_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(original_dir)

    # Create the `augmented_directory` in the parent directory with a subfolder named like the current folder
    augmented_directory = os.path.join(parent_dir, 'augmented_directory', os.path.basename(original_dir))
    os.makedirs(augmented_directory, exist_ok=True)

    # Apply augmentations
    augmentations = {
        "_flip": ImageOps.mirror(original_image),
        "_rotate": original_image.rotate(90, expand=True),
        "_blur": original_image.filter(ImageFilter.BLUR),
        "_contrast": ImageOps.autocontrast(original_image),
        "_crop": original_image.crop((
            original_image.width / 4, original_image.height / 4,
            3 * original_image.width / 4, 3 * original_image.height / 4
        )),
        "_illuminate": ImageOps.solarize(original_image)
    }

    # Save each augmented image in both the original folder and `augmented_directory`
    for suffix, img in augmentations.items():
        # Path for the original directory
        save_path_original = f'{base}{suffix}{extension}'
        img.save(save_path_original)

        # Path for `augmented_directory`
        save_path_augmented = os.path.join(augmented_directory, f'{os.path.basename(base)}{suffix}{extension}')
        img.save(save_path_augmented)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Apply various augmentations to an image.")
	parser.add_argument("file_path", type=str, help="Path to the image file to be augmented")
	args = parser.parse_args()

	apply_augmentations(args.file_path)