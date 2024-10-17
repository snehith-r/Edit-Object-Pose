from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting
import cv2
import numpy as np
from utils.config_parser import load_config


def inpaint(image, mask, prompt,negative_prompt, guidance_scale,output_path):
    # Load the Stable Diffusion inpainting model
    # model_name = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    model_name = "stabilityai/stable-diffusion-2-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    # pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    # Define the kernel size for dilation (increase to dilate more)
    kernel = np.ones((5, 5), np.uint8)

    # Apply dilation to the mask
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)
    cv2.imwrite("dilated_mask.png", dilated_mask)

    # Load the original image (with object removed) and mask
    original_image = Image.open(image).convert("RGB")
    mask_image = Image.open("dilated_mask.png").convert("L")  # Mask image indicating where to inpaint

    # Set the prompt for guidance, if necessary
    # prompt = "Inpaint the masked area to seamlessly match the background scene . It should be coherent and smooth."
    

    # prompt = "fill to blend with the incomplete background room. make it consistent and coherent with the scene."

    # prompt="fill the mask as a continuation of wall and floor to complete the scene. strictly no new entities or objects."
    # negative_prompt ="not wall, not floor,art,add objects,add partial objects,new objects in scene,chair extended"
    # guidance_scale=8

    # prompt = "Restore the missing parts of the wall and floor, smooth and clean surfaces to complete the scene"
    # prompt ="Strictly restore the missing parts of white wall, floor, borders of a single chair sofa facing sideways to complete the scene. high resolution, smooth sharp and consistent"
    # negative_prompt = "artifacts,chair,sofa,furniture"
    # guidance_scale = 8  # reduce the guidance to make it less strict on details

    # prompt ="Strictly restore the missing parts of white sofa in the background, floor and table to complete the scene."
    # negative_prompt = "artifacts, table,furniture"
    # guidance_scale = 6  # reduce the guidance to make it less strict on details

    # prompt ="Strictly restore the missing parts of wall, floor,table, and laptop to complete the scene. high resolution, smooth sharp and consistent"
    # negative_prompt = "artifacts,office chair,chair furniture"
    # guidance_scale = 6  # reduce the guidance to make it less strict on details
    
    # Run the inpainting model
    result = pipe(prompt=prompt, 
                    image=original_image, 
                    mask_image=mask_image, 
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale,num_inference_steps=50,
                    ).images[0]

    # Save the result
    result.save(output_path)

# config = load_config("config.yaml")

#inpaint(config["pose_edited_image"],config["inpaint_mask"],config["prompt"],config["negative_prompt"],config["guidance_scale"],config["inpaint_output_image"])