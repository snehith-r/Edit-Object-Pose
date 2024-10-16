import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
chair_image = cv2.imread('table.jpg')
mask = cv2.imread('results/table_a-63_p25/segmentation_mask.png', cv2.IMREAD_GRAYSCALE)
new_object = cv2.imread("view_synthesis/segmented_rgba.png@20241016-183629/save/it0-test/0.png")
new_object_mask = cv2.imread('results/table_a-63_p25/pose_rotated_object_mask.png', cv2.IMREAD_GRAYSCALE)  # Provided segmentation mask

subtracted_mask = cv2.subtract(mask, new_object_mask)

combined_mask = cv2.bitwise_or(mask, new_object_mask)

# Save the combined mask (optional)
cv2.imwrite("combined_mask.png", combined_mask)

# Step 1: Remove the Chair using the original mask
inverted_mask = cv2.bitwise_not(combined_mask)

# Extract the background using the inverted mask
background = cv2.bitwise_and(chair_image, chair_image, mask=inverted_mask)

cv2.imwrite('background.png', background)


# Step 2: Save the subtracted mask for inpainting
cv2.imwrite('subtracted_mask.png', subtracted_mask)
exit(0)

# Step 1: Remove the Chair using the original mask
inverted_mask = cv2.bitwise_not(mask)

# Extract the background using the inverted mask
background = cv2.bitwise_and(chair_image, chair_image, mask=inverted_mask)

# Step 2: Resize the new object and its mask to match the background size
new_object_resized = cv2.resize(new_object, (background.shape[1], background.shape[0]))
new_object_mask_resized = cv2.resize(new_object_mask, (background.shape[1], background.shape[0]))

# Step 3: Extract the new object using the provided mask
new_object_extracted = cv2.bitwise_and(new_object_resized, new_object_resized, mask=new_object_mask_resized)

# Step 4: Replace the background pixels with the non-white pixels from the new object
final_image = background.copy()
final_image[new_object_mask_resized == 255] = new_object_extracted[new_object_mask_resized == 255]

# Step 5: Save the final image
cv2.imwrite('final_output_with_segmented_mask.png', final_image)

# Plot the results for visualization
plt.figure(figsize=(10, 10))

# Display the extracted background (after chair removal)
plt.subplot(2, 2, 1)
plt.title("Extracted Background (Chair Removed)")
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display the provided new object mask
plt.subplot(2, 2, 2)
plt.title("Provided New Object Mask")
plt.imshow(new_object_mask_resized, cmap='gray')
plt.axis('off')

# Display the extracted new object using the provided mask
plt.subplot(2, 2, 3)
plt.title("Extracted New Object with New Mask")
plt.imshow(cv2.cvtColor(new_object_extracted, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display the final output image with the new object on the background
plt.subplot(2, 2, 4)
plt.title("Final Image with New Object on Background")
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

