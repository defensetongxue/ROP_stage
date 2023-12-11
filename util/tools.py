from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import numpy as np
from random import shuffle
def judge(mask, left_, upper_, right_, lower_, threshold):
    """
    Determine if the sum of values in a specified region of a mask exceeds a threshold.
    
    Args:
    mask (np.array): A numpy array representing the mask.
    left (int): Left boundary of the region.
    upper (int): Upper boundary of the region.
    right (int): Right boundary of the region.
    lower (int): Lower boundary of the region.
    threshold (float): Threshold to compare against the sum of the region.

    Returns:
    bool: True if the sum of the region exceeds the threshold, else False.
    """
    # Extract the specified region from the mask
    region = mask[upper_:lower_,left_:right_]
    # print(region.shape, np.sum(region),left_, right_, upper_, lower_)
    # Sum the values in the region
    region_sum = np.sum(region)

    # Check if the sum exceeds the threshold
    return region_sum >= threshold

def crop_patches(img,patch_size,x,y,
                 abnormal_mask,  # abnormal means stage 3 abnormal
                 stage,max_weight=1599,max_height=1199, check_padding=20,save_dir=None,image_cnt='1'):
    '''
    const resize_height=600
    const resize_weight=800
    keep the size as conv ridge segmentation model
    '''
    left = x - patch_size // 2
    upper = y - patch_size // 2
    right = x + patch_size // 2
    lower = y + patch_size // 2
    # Pad if necessary
    padding = [max(0, -left), max(0, -upper), max(0, right - 800), max(0, lower - 600)]
    left=max(0, left)
    upper=max(0,upper)
    right=min(max_weight, right)
    lower= min(max_height, lower)
    # Crop the patch
    patch = img.crop((left, upper, right, lower))
    patch = ImageOps.expand(patch, tuple(padding), fill=255) 
    res = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
    # Create a circular mask for the inscribed circle
    mask = Image.new('L', (patch_size, patch_size), 0)
    draw = ImageDraw.Draw(mask)
    radius = patch_size // 2
    center = (radius, radius)
    draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
    res.paste(patch, (0, 0), mask=mask)
    
    if stage==3:
        assert abnormal_mask is not None
        if judge(abnormal_mask,left+check_padding,upper+check_padding,right-check_padding,lower-check_padding,1):
            patch_stage=3
        else:
            patch_stage=2
    else:
       patch_stage=stage
    
    save_path=os.path.join(save_dir,f"{str(patch_stage)}_{image_cnt}.jpg")
    res.save(save_path)
    return patch_stage
    
def sample_patch(ridge_diffusion_path,sample_dense,max_sample_number):
    ridge_mask=Image.open(ridge_diffusion_path).convert('L')
    ridge_mask=np.array(ridge_mask).squeeze()
    ridge_mask[ridge_mask!=0]=1
    sample_list = []
    
    while True:
        # this line will only return the first idx that equal to 1
        idx = np.unravel_index(np.argmax(ridge_mask, axis=None), ridge_mask.shape)

        if ridge_mask[idx]<1:
            break

        # Clear the square region around the point
        x, y = idx[0], idx[1]
        sample_list.append([y,x]) # in format weight_coor, height_coor always
        xmin, ymin = max(0, x -sample_dense), max(0, y - sample_dense)
        xmax, ymax = min(ridge_mask.shape[0], x +sample_dense), min(ridge_mask.shape[1], y + sample_dense)
        ridge_mask[ xmin:xmax,ymin:ymax] = -9
    if len(sample_list)>max_sample_number:
        shuffle(sample_list)
        sample_list=sample_list[:max_sample_number]
    sample_list=np.array(sample_list,dtype=np.float16)
    
    return sample_list

def visual_sentence(image_path, x, y, patch_size, label=1, confidence=0.0, text=None, save_path=None, font_size=20):
    # Open the image and resize
    img = Image.open(image_path).resize((800, 600))

    # Set the box color based on the label
    box_color = 'green' if label == 1 else 'yellow' if label == 2 else 'red'

    # Calculate the top-left and bottom-right coordinates of the box
    half_size = patch_size // 2
    top_left_x = x - half_size
    top_left_y = y - half_size
    bottom_right_x = x + half_size
    bottom_right_y = y + half_size

    # Draw the box
    draw = ImageDraw.Draw(img)
    draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline=box_color, width=3)

    # Draw the confidence value near the top left of the box
    confidence_text = f"{confidence:.2f}"  # Format confidence to 2 decimal places
    draw.text((top_left_x, top_left_y - font_size-2), confidence_text, fill=box_color, font=ImageFont.truetype("./arial.ttf", font_size))

    # Draw the additional text if provided
    if text is not None:
        # Load the Arial font with the specified font size
        font = ImageFont.truetype("./arial.ttf", font_size)
        text_position = (10, 10)  # Top left corner
        draw.text(text_position, text, fill="white", font=font)

    # Save or show the image
    if save_path:
        img.save(save_path)
    else:
        img.show()