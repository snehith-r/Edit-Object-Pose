from PIL import Image
import numpy as np
import cv2
import numpy as np

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(image, path):
    """
    Convert a NumPy array to a PIL image and save it.
    Handles single-channel and multi-channel images.
    """

    # Save the image
    image.save(path)

def overlay(orig_path,mask_path,rotated_mask_path,rotated_image_path,inpaint_mask_path,pose_edited_path):
    # Load the images
    orig_image = cv2.imread(orig_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    new_object = cv2.imread(rotated_image_path)
    new_object_mask = cv2.imread(rotated_mask_path, cv2.IMREAD_GRAYSCALE)  # Provided segmentation mask

    subtracted_mask = cv2.subtract(mask, new_object_mask)

    # Step 2: Save the subtracted mask for inpainting
    cv2.imwrite(inpaint_mask_path, subtracted_mask)

    # Step 1: Remove the Chair using the original mask
    inverted_mask = cv2.bitwise_not(mask)

    # Extract the background using the inverted mask
    background = cv2.bitwise_and(orig_image, orig_image, mask=inverted_mask)

    # Step 2: Resize the new object and its mask to match the background size
    new_object_resized = cv2.resize(new_object, (background.shape[1], background.shape[0]))
    new_object_mask_resized = cv2.resize(new_object_mask, (background.shape[1], background.shape[0]))

    # Step 3: Extract the new object using the provided mask
    new_object_extracted = cv2.bitwise_and(new_object_resized, new_object_resized, mask=new_object_mask_resized)

    # Step 4: Replace the background pixels with the non-white pixels from the new object
    final_image = background.copy()
    final_image[new_object_mask_resized == 255] = new_object_extracted[new_object_mask_resized == 255]

    # Step 5: Save the final image
    cv2.imwrite(pose_edited_path, final_image)
