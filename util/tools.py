from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
import numpy as np

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

def crop_patches(image_path,
                 ridge_diffusion_path,
                 abnormal_cooridinate,  # abnormal means stage 3 abnormal
                 stage,
                 point_list, # point list is parse from torch.tensor so in height,width format
                 value_list, 
                 word_number=5, patch_size=16, save_path=None,
                 lower_bound=0.4):
    '''
    const resize_height=600
    const resize_weight=800
    keep the size as conv ridge segmentation model
    '''
    # Load image
    img = Image.open(image_path)

    # Resize image
    img = img.resize((800,600))

    # build ridge_diffusion_map
    if ridge_diffusion_path is not None:
        ridge_mask=Image.open(ridge_diffusion_path).resize((800,600))
        ridge_mask=np.array(ridge_mask)
        ridge_mask[ridge_mask>0]=1
    else:
        ridge_mask=None
    # build abnormal_mask
    if abnormal_cooridinate is not None:
        abnormal_mask=np.zeros((600,800), dtype=np.uint8)
        for y,x in abnormal_cooridinate:
            abnormal_mask[int(x/2),int(y/2)]=1
    else:
        abnormal_mask=None
    stage_list=[]
    cnt=0
    # Prepare patches
    for( y, x),val in zip(point_list[:word_number],value_list[:word_number]):  # Use only first 'word_number' points
        # Scale points according to resized image
        if val < lower_bound:
            break
            # create a zero image in jpg format and save in 
            patch = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
            patch.save(os.path.join(save_path,f"{str(cnt)}_0.jpg"))
            cnt+=1
            stage_list.append(0)
            break
        left = x - patch_size // 2
        upper = y - patch_size // 2
        right = x + patch_size // 2
        lower = y + patch_size // 2

        # Pad if necessary
        padding = [max(0, -left), max(0, -upper), max(0, right - 800), max(0, lower - 600)]
        left=max(0, left)
        upper=max(0,upper)
        right=min(800, right)
        lower= min(600, lower)
        # Crop the patch
        patch = img.crop((left, upper, right, lower))
        patch = ImageOps.expand(patch, tuple(padding), fill=255) 
        # Create a black image for the background
        res = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))

        # Create a circular mask for the inscribed circle
        mask = Image.new('L', (patch_size, patch_size), 0)
        draw = ImageDraw.Draw(mask)
        radius = patch_size // 2
        center = (radius, radius)
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), fill=255)
        # print(mask.size,patch.size)
        # Paste the patch onto the black background using the mask
        res.paste(patch, (0, 0), mask=mask)
        cnt += 1
        if ridge_mask is None or \
            not judge(ridge_mask,left,upper,right,lower,1):
            patch_stage=0
        elif stage==3:
            assert abnormal_mask is not None
            if judge(abnormal_mask,left,upper,right,lower,1):
                patch_stage=3
            else:
                patch_stage=2
        else:
            patch_stage=stage
        
        stage_list.append(patch_stage)
        if patch_stage>0:
            res.save(os.path.join(save_path, f"{cnt}_{str(patch_stage)}.jpg"))
    return stage_list
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