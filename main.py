# main.py

import yaml
from segmentation import sam_segmentation
from pose_editing import zero123_pose_editing
from inpainting import stable_diffusion_inpainting
from utils.image_utils import load_image, save_image, overlay
from utils.config_parser import load_config
from entity_extraction_llm import extract_entities_llm  # Import the LLM-based extractor
from PIL import Image
import numpy as np

import os
import glob

def find_latest_0_png(folder_path):
    # Search for all '0.png' files recursively in the folder and subfolders
    file_paths = glob.glob(os.path.join(folder_path, '**', '0.png'), recursive=True)
    
    # Check if any '0.png' files were found
    if not file_paths:
        return None
    
    # Find the latest '0.png' file based on modification time
    latest_file = max(file_paths, key=os.path.getmtime)
    
    return latest_file

def main():
    # Load configuration
    config = load_config("config.yaml")

    

    # Extract entities from user input using LLM
    user_input = config.get("user_input", "")
    quantize = config.get("quantize", False)  # Optional quantization flag

    object_name, azimuth, polar = extract_entities_llm(user_input, quantize=quantize)

    if object_name is None or azimuth is None or polar is None:
        print("Failed to extract entities from user input.")
        return

    # Step 1: Segmentation
    segmentation_method = config['segmentation']['method']
    image = load_image(config['input_image'])

    if segmentation_method == "sam":
        mask,mask_image = sam_segmentation.segment(image, f"a {object_name}.")
    
    else:
        print("Invalid segmentation method specified.")
        return

    save_image(mask_image, config['mask_image'])

    # Convert the image to RGBA
    rgba_image = image.convert('RGBA')

    # Load the mask image (assuming it's a grayscale image where mask pixels are non-zero)
    mask_image = Image.open(config['mask_image']).convert('L')

    # Convert mask to numpy array
    mask_array = np.array(mask_image)

    # Convert RGBA image to numpy array
    rgba_array = np.array(rgba_image)

    # Set alpha channel to 255 where mask is non-zero
    rgba_array[:, :, 3] = np.where(mask_array != 0, 255, rgba_array[:, :, 3])
    rgba_array[:, :, 3] = np.where(mask_array == 0, 0, rgba_array[:, :, 3])


    # Convert back to PIL Image
    masked_rgba_image = Image.fromarray(rgba_array)


    masked_rgba_image.save(config['segmented_image'])


    # Step 2: Pose Editing
    pose_editing_method = config['pose_editing']['method']

    if pose_editing_method == "zero123":
        edited_image = zero123_pose_editing.edit_pose('../'+config['segmented_image'], azimuth, polar)
    elif pose_editing_method == "clip_gan":
        edited_image = clip_gan_pose_editing.edit_pose(image, mask, azimuth, polar)
    elif pose_editing_method == "nerf":
        edited_image = nerf_pose_editing.edit_pose(image, mask, azimuth, polar)
    else:
        print("Invalid pose editing method specified.")
        return

    folder_path= 'view_synthesis'
    
    stablezero123_out_path = find_latest_0_png(folder_path)
    mask, mask_image = sam_segmentation.segment(Image.open(stablezero123_out_path).convert("RGB"), f'a {object_name}')
    mask_image.save(config['pose_rotated_mask'])

    overlay(config['input_image'],
            config['mask_image'],
            config['pose_rotated_mask'],
            stablezero123_out_path,
            config['inpaint_mask'],
            config['pose_edited_image']
            )

    # Step 3: Inpainting
    inpainting_method = config['inpainting']['method']

    if inpainting_method == "stable_diffusion":
        final_image = stable_diffusion_inpainting.inpaint(config["pose_edited_image"],config["inpaint_mask"],config["prompt"],config["negative_prompt"],config["guidance_scale"],config["inpaint_output_image"])
    

    # Save the final output
    # save_image(final_image, config['output_image'])

if __name__ == "__main__":
    main()
