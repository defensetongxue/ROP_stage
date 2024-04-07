from PIL import Image, ImageDraw, ImageFont
import os
import json

# Load the necessary data
with open('./confuse_list.json') as f:
    confuse_list = json.load(f)

# Specify font
font_path = './arial.ttf'
font = ImageFont.truetype(font_path, 70)

# Ensure the output directory exists
output_dir = './experiments/release_check'
os.makedirs(output_dir, exist_ok=True)
for stage_list in ['0','1','2','3']:
    os.makedirs(output_dir+'/'+stage_list, exist_ok=True)
# Process each image
for image_name, details in confuse_list.items():
    image_path = details["image_path"]
    try:
        # Open the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Get image dimensions
        width, height = image.size

        # Define text positions
        positions = [(0, 0), (width, 0), (0, height), (width, height)]
        keys = ["annote", "xsj", "zy", "model_prediction"]

        # Draw text in four corners
        for pos, key in zip(positions, keys):
            text = f"{key}: {str(confuse_list[image_name][key])}"
            # Calculate text size using textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            # Adjust text position to not overlap the image borders
            adjusted_pos = (max(pos[0] - text_width, 0), max(pos[1] - text_height, 0))
            draw.text(adjusted_pos, text, font=font, fill="white")

        # Save the image
        save_path = os.path.join(output_dir,str(confuse_list[image_name]["annote"] ),image_name)
        image.save(save_path)
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
