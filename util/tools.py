from PIL import Image,ImageOps

def crop_patches(image_path, point_list, resize_height=224, word_number=5, patch_size=16, save_path=None):
    # Load image
    img = Image.open(image_path)
    orig_width, orig_height = img.size

    # Calculate resize width while keeping aspect ratio
    aspect_ratio = orig_width / orig_height
    resize_width = int(resize_height * aspect_ratio)

    # Resize image
    img = img.resize((resize_width, resize_height))

    # Prepare patches
    patches = []
    for point in point_list[:word_number]:  # Use only first 'word_number' points
        x, y = point
        left = x - patch_size // 2
        upper = y - patch_size // 2
        right = x + patch_size // 2
        lower = y + patch_size // 2

        # Pad if necessary
        padding = [max(0, -left), max(0, -upper), max(0, right - resize_width), max(0, lower - resize_height)]
        patch = img.crop((max(0, left), max(0, upper), min(resize_width, right), min(resize_height, lower)))
        patch = ImageOps.expand(patch, tuple(padding), fill=5)  # Fill value 5 for padding

        patches.append(patch)

    # Add zero images if fewer points than word_number
    while len(patches) < word_number:
        patches.append(Image.new('RGB', (patch_size, patch_size), (0, 0, 0)))

    # Concatenate patches
    concatenated_image = Image.new('RGB', (patch_size * word_number, patch_size))
    for i, patch in enumerate(patches):
        concatenated_image.paste(patch, (i * patch_size, 0))

    # Save or return the concatenated image
    if save_path:
        concatenated_image.save(save_path)
    
    return concatenated_image
