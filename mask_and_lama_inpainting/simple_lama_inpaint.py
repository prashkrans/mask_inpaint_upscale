import os
from simple_lama_inpainting import SimpleLama
from utils import output_dir

def simple_lama_inpaint(images, masks, img_names_with_ext):

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for image, mask, img_name_with_ext in zip(images, masks, img_names_with_ext):
        img_name = img_name_with_ext.split('.')[0]
        output_name = f'inpaint_{img_name}.png' # Input could be .jpg or .png but saving as .png (as png is a loss less format)
        output_path = f'{output_dir}/{output_name}'
        print(f'Initiating simple lama inpainting for the obtained image {img_name_with_ext} and its mask')
     
        mask = mask.convert('L') # Convert the mask image to grayscale

        simple_lama = SimpleLama()
        result = simple_lama(image, mask)
        result.save(f"{output_path}") 

    print(f'Inpaiting task completed for all {len(images)} images')
