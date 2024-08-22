#!/bin/bash

source env_m_i_u/bin/activate

cd mask_and_lama_inpainting/
rm -rf input_mask descaled_image # # Deleting input_mask and descaled_image directories is important as we need to inpaint and upscale newer images only
python3 main_mask_gen_descale_inpaint.py

echo 'BSRGAN processing started'
cd ../BSRGAN/
echo "Changed directory to : ${PWD}"
python3 main_bsrgan.py

echo "The entire flow was completed successfully"

# Delete the intermediary inpaint output that is of lower resolution
rm -rf ../mask_and_lama_inpainting/output_image
deactivate
