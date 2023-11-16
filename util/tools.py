from PIL import Image, ImageOps
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
# Example usage
# crop_patches("path_to_image.jpg", original_width, original_height, [(50, 50), (100, 100)], save_path="output.jpg")
