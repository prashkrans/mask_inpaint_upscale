import os
import json

with open('config.json') as f:
    config = json.load(f)

img_dir = config['IMG_DIR']
mask_dir = config['MASK_DIR']
output_dir = config['OUTPUT_DIR']
descaled_dir = config['DESCALED_DIR']

def get_image_names_from_dir(dir):
    # List files in the directory
    files = os.listdir(dir)
    img_names = []

    for file in files:
        # Check if the file is a .jpg or .png file
        if file.lower().endswith(('.jpg', '.png', 'jpeg')):
            img_names.append(file)
    
    return img_names
