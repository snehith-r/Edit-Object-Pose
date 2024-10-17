import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_bounding_box(image, bounding_box):
    # Create a plot with the image
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Add the bounding box as a rectangle
    box = patches.Rectangle(
        (bounding_box[0], bounding_box[1]),
        bounding_box[2] - bounding_box[0],
        bounding_box[3] - bounding_box[1],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(box)
    
    plt.show()

def segment(image, prompt):
    # Load Grounding DINO for bounding box detection
    model_id = "IDEA-Research/grounding-dino-tiny"
    dino_processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda")
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    
    
    # Prepare the image and prompt for grounding
    inputs = dino_processor(images=image, text=[prompt], return_tensors="pt").to("cuda")

    # Generate bounding boxes from Grounding DINO
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process the bounding boxes (post-processing to get box predictions)
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]  # target size is image size in (height, width)
    )

    # Retrieve the first bounding box (you can loop if there are multiple boxes)
    if len(results[0]["boxes"]) > 0:
        bounding_box = results[0]["boxes"][0].cpu().numpy()
    else:
        raise ValueError("No bounding box detected for the given prompt")

    # visualize_bounding_box(image, bounding_box)
    print(bounding_box)

    boxes = [[bounding_box.tolist()]]

    
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)
    
    # Convert the mask to a NumPy array and save it
    mask_image = Image.fromarray(masks[0]*255)
    
    return masks[0], mask_image

# Example usage:

# image_path = 'view_synthesis/segmented.png@20241015-044620/save/it0-test/0.png'
# mask, mask_image = segment(Image.open(image_path).convert("RGB"), 'a chair')
# mask_image.save('new_object_mask.png')

