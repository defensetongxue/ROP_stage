from PIL import Image, ImageOps, ImageDraw, ImageFont
import os
        
def crop_patches(image_path, point_list, word_number=5, patch_size=16, save_path=None):
    '''
    const resize_height=600
    const resize_weight=800
    keep the size as conv ridge segmentation model
    '''
    # Load image
    img = Image.open(image_path)

    # Resize image
    img = img.resize((800,600))

    # Scale factor for coordinates
    cnt=0
    # Prepare patches
    for y, x in point_list[:word_number]:  # Use only first 'word_number' points
        # Scale points according to resized image
        
        left = x - patch_size // 2
        upper = y - patch_size // 2
        right = x + patch_size // 2
        lower = y + patch_size // 2

        # Pad if necessary
        padding = [max(0, -left), max(0, -upper), max(0, right - 800), max(0, lower - 600)]
        patch = img.crop((max(0, left), max(0, upper), min(800, right), min(600, lower)))
        patch = ImageOps.expand(patch, tuple(padding), fill=255)  # Fill value 5 for padding

        patch.save(os.path.join(save_path,f"{str(cnt)}.jpg"))
        cnt+=1

def visual_sentence(image_path, x, y, patch_size, label=1, text=None, save_path=None, font_size=20):
    assert label in [1, 2, 3], label

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

    # Draw the text if provided
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