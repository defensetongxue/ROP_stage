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

def crop_patches(img,patch_size,x,y,
                 abnormal_mask,  # abnormal means stage 3 abnormal
                 stage,max_weight=1599,max_height=1199, check_padding=50,save_dir=None,image_cnt='1'):
    '''
    keep the size as conv ridge segmentation model
    '''
    left  = int(x - patch_size // 2)
    upper = int(y - patch_size // 2)
    right = int(x + patch_size // 2)
    lower = int(y + patch_size // 2)
    
    # Pad if necessary
    padding = [max(0, -left), max(0, -upper), max(0, right - max_weight), max(0, lower - max_height)]
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
            # in that case if we can ensure the other part of the ridge is stage 2?
            if True: # if we ca
                return -1,res
    else:
       patch_stage=stage
    if save_dir:
        save_path=os.path.join(save_dir,f"{str(patch_stage)}_{image_cnt}.jpg")
        res.save(save_path)
    return patch_stage,res
    

def sample_patch(ridge_diffusion_path, sample_dense, max_sample_number):
    ridge_mask = Image.open(ridge_diffusion_path).convert('L')
    ridge_mask = np.array(ridge_mask).squeeze()
    ridge_mask[ridge_mask != 0] = 1

    sample_list = []
    while len(sample_list) < max_sample_number:
        y_indices, x_indices = np.where(ridge_mask == 1)

        # Break if no more points to sample
        if len(y_indices) == 0:
            break

        # Randomly select one of the points
        idx = np.random.choice(len(y_indices))
        y, x = y_indices[idx], x_indices[idx]

        # Add the point to the sample list
        sample_list.append([int(x), int(y)])  # format: width_coord, height_coord
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
            # create a zero image in jpg format and save in 
            patch = Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
            patch.save(os.path.join(save_path,f"{str(cnt)}.jpg"))
            cnt+=1
            stage_list.append(0)
            continue
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
        res.paste(patch, (0, 0), mask=mask)
        res.save(os.path.join(save_path, f"{cnt}.jpg"))
        cnt += 1
        if ridge_mask is None or \
            not judge(ridge_mask,left,upper,right,lower,1):
            stage_list.append(0) # ridge in this patch
            continue
        if stage==3:
            assert abnormal_mask is not None
            if judge(abnormal_mask,left,upper,right,lower,1):
                stage_list.append(3)
            else:
                stage_list.append(2)
            continue
        stage_list.append(stage)
    return stage_list
def visual_sentence(image_path, x, y, patch_size, label=1, confidence=0.0, text=None, save_path=None, font_size=20):
    # Open the image and resize
    img = Image.open(image_path).resize((800, 600))

    # Set the box color based on the label
    box_color = 'green' if label == 1 else 'yellow' if label == 2 else 'red'

        # Clear the square region around the point
        xmin, xmax = max(0, x - sample_dense), min(ridge_mask.shape[1], x + sample_dense + 1)
        ymin, ymax = max(0, y - sample_dense), min(ridge_mask.shape[0], y + sample_dense + 1)
        ridge_mask[ymin:ymax, xmin:xmax] = 0

    # sample_list = np.array(sample_list, dtype=np.float16)
    return sample_list


def visual_sentences(image_path, points, patch_size, label=None, confidences=None, text=None, save_path=None, font_size=60,sample_visual=[]):
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    box_color = 'green' if label == 1 else 'yellow' if label == 2 else 'red'
    # Iterate over each point
    for i, (x, y) in enumerate(points):

        # Set the box color based on the label

        # Calculate the top-left and bottom-right coordinates of the box
        half_size = patch_size // 2
        top_left_x = x - half_size
        top_left_y = y - half_size
        bottom_right_x = x + half_size
        bottom_right_y = y + half_size

        # Draw the box
        draw.rectangle([top_left_x, top_left_y, bottom_right_x, bottom_right_y], outline=box_color, width=3)

        # Draw the confidence value near the top left of the box
        confidence_text = f"{confidences[i]:.2f}"  # Format confidence to 2 decimal places
        draw.text((top_left_x, top_left_y - font_size - 2), confidence_text, fill=box_color, font=ImageFont.truetype("./arial.ttf", font_size))

    # Draw additional text if provided
    if text is not None:
        # Load the Arial font with the specified font size
        font = ImageFont.truetype("./arial.ttf", font_size)
        text_position = (10, 10)  # Top left corner
        draw.text(text_position, text, fill="white", font=font)
    for x, y in sample_visual:
        # Define the position for the star
        star_position = (x, y)

        # Define the star symbol and its color
        star_symbol = "*"
        star_color = "blue"  # You can choose any color you like

        # Draw the star on the image
        draw.text(star_position, star_symbol, fill=star_color, font=ImageFont.truetype("./arial.ttf", font_size))
    # Save or show the image
    if save_path:
        img.save(save_path)
    else:
        img.show()
        
def k_max_values_and_indices(scores, k,r=100,threshold=0.0):
    # Flatten the array and get the indices of the top-k values

    preds_list = []
    maxvals_list = []

    for _ in range(k):
        idx = np.unravel_index(np.argmax(scores, axis=None), scores.shape)

        maxval = scores[idx]
        if maxval<threshold:
            break
        maxvals_list.append(float(maxval))
        preds_list.append(idx)
        # Clear the square region around the point
        x, y = idx[0], idx[1]
        xmin, ymin = max(0, x - r // 2), max(0, y - r // 2)
        xmax, ymax = min(scores.shape[0], x + r // 2), min(scores.shape[1], y + r // 2)
        scores[ xmin:xmax,ymin:ymax] = -9
    maxvals_list=np.array(maxvals_list,dtype=np.float16)
    preds_list=np.array(preds_list,dtype=np.float16)
    return maxvals_list, preds_list