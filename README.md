# Mask Inpaint Upscale
Inpainting model used: Lama Inpainting  
Upscaling models used: Real-ESRGAN and BSRGAN

### Prerequisites:
- Python 3.10 (Might work with higher versions as well)
- Nvidia drivers installed along with CUDA
- Nvidia GPU with VRAM atleast 6GB (Real-ESRGAN) or 12GB (BSRGAN)
- CPU can also be used but is much slower

### Setup:
```
git clone https://github.com/prashkrans/mask_inpaint_upscale.git
cd mask_inpaint_upscale/

# Create a new python virtual environment. Make sure to name the env as below. Any other name would break the script. 
python3 -m venv env_m_i_u
source env_m_i_u/bin/activate

# Install the required dependencies
pip install torch==1.8.0 torchvision==0.9.0
pip install -r requirements.txt
```
### Usage:
1. Perform inpainting using Lama, then upscale using both Real-ESRGAN and BSRGAN:
```
# Put the image(s) in the ./initial_input folder (multiple images can be processed at once)

# Make the script executable (need to run the below command only once)
chmod +x run_entire_flow_4x.sh 

# Run the bash script. Please go through it once to assure yourself that its safe to run this script
./run_entire_flow_4x.sh

# Images pop up sequentially asking user input to paint a mask i.e. the region over the image that needs to be removed. 

# Use "=" or "up arrow" to increase the brush size and "-" or "down arrow" to decrease the brush size

# Press "Escape" to revert all the changes and repaint the mask

# Finally press "Enter" to save the mask(s) and wait for processing to complete

# Real-ESRGAN upscaled output is saved in the ./final_output/esrgan_output while BSRGAN upscaled output is saved int the ./final_output/bsrgan_output

# Generally, BSRGAN results are better than that of Real-ESRGAN
```

2. Perform inpainting using Lama, then upscale using Real-ESRGAN with the following scaling factors: 1x, 2x, 3x, and 4x:
```
# Put the image(s) in the ./initial_input folder (multiple images can be processed at once)

# Make the script executable (need to run the below command only once)
chmod +x run_entire_flow_with_RealESRGAN_1_to_4.sh 

# Run the bash script. Please go through it once to assure yourself that its safe to run this script
./run_entire_flow_with_RealESRGAN_1_to_4.sh

# Images pop up sequentially asking user input to paint a mask i.e. the region over the image that needs to be removed. 

# Use "=" or "up arrow" to increase the brush size and "-" or "down arrow" to decrease the brush size

# Press "Escape" to revert all the changes and repaint the mask

# Finally press "Enter" to save the mask(s) and wait for processing to complete

# Real-ESRGAN upscaled output is saved in the ./final_output/esrgan_output
```

3. Perform upscaling using Real-ESRGAN only, without inpainting: 
```
# Put the image(s) in the ./initial_input/upscaler_input_dir folder (multiple images can be processed at once)

# Make the script executable (need to run the below command only once)
chmod +x run_RealESRGAN_upscaler_only.sh

# Run the bash script. Please go through it once to assure yourself that its safe to run this script
./run_RealESRGAN_upscaler_only.sh

# Real-ESRGAN upscaled output is saved in the ./final_output/esrgan_output
```

When the .sh file is run the first time it automatically downloads two models about a GB in size. 

### Flow:
1. Takes input of image(s) from the user and then descales them to avoid CUDA OOM errors.
2. Previews the image(s) sequentially in a 1280x1280 window.
3. Allows the user to select a mask on the image using the mouse (translucent black color).
4. Saves the mask(s) in black and white at the same size as the image.
5. Uses the image(s) along with their mask(s) in LAMA inpainting to remove objects within the selected region in the mask(s) and merges them with the surroundings. No prompt is required.
6. Upscales the inpainted image(s) up to 4x using Real-ESRGAN and BSRGAN upscaling models and saves the upscaled images.

#### Step 1. Create mask:
User is prompted to paint a mask over the provided image using mouse.

#### Step 2. Simple Lama Inpainting:
https://github.com/enesmsahin/simple-lama-inpainting

#### Note:
- Since 6GB VRAM is not enough, high resolution images are descaled to min (height, width) = 1024 (not 512 - even though inpainting is best at 512 but upscaling later is not good)
For 1024 px - inpainting is alright but upscaling is really good hence, 1024 px is preferred over 512 px

Generally a descale_factor of 0.15 is sufficient for descaling 4000x6000 px^2 to acceptable low res image
Calculation of descale factor is already taken care of in the code (custom calc)

After descaling, mask is created, then both the descaled image and its mask are fed to simple lama inpainting which generates an inpainted but low res image.

#### Step 3. Upscaler 4x:
Upscales the lower resolution inpainted image upto 4x using the models below: 
1. [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
1. [BSRGAN](https://github.com/cszn/BSRGAN) (Slightly Better)

### Troubleshooting:
Error - 
File "/home/nashprat/.local/lib/python3.10/site-packages/basicsr/data/degradations.py", line 8, in <module>
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'

Error fix:
Line 8 of /home/nashprat/workspace/real_ESRGAN/esrgan_env/lib/python3.10/site-packages/basicsr/data/degradations.py
from torchvision.transforms.functional_tensors import rgb_to_grayscale
to
from torchvision.transforms.functional import rgb_to_grayscale

Finally, the low res inpainted image is upscaled 4x times by Real-ESRGAN

Usage: `python inference_realesrgan.py -n RealESRGAN_x4plus -i infile -o outfile [options]`...

### Credits:
1. [Lama Inpainting](https://github.com/advimman/lama) | [Apache-2.0 License](https://github.com/advimman/lama?tab=License-1-ov-file)
2. [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | [BSD-3-Clause license](https://github.com/xinntao/Real-ESRGAN?tab=BSD-3-Clause-1-ov-file)
2. [BSRGAN](https://github.com/cszn/BSRGAN) | [Apache-2.0 License](https://github.com/cszn/BSRGAN?tab=Apache-2.0-1-ov-file)
