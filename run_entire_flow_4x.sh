#!/bin/bash

source env_m_i_u/bin/activate

cd mask_and_lama_inpainting/
rm -rf input_mask descaled_image # # Deleting input_mask and descaled_image directories is important as we need to inpaint and upscale newer images only
python3 main_mask_gen_descale_inpaint.py

cd ../Real-ESRGAN/

echo "Changed directory to : ${PWD}"

# for i in {1..4}; do
#     python3 inference_realesrgan.py -n RealESRGAN_x4plus -i ../mask_and_lama_inpainting/output_image -o ../final_output --face_enhance -s $i
# done

echo 'RealESRGAN processing started'
# Check if the ../final_output/esrgan_output directory exists, if not, create it
if [ ! -d "../final_output/esrgan_output" ]; then
  mkdir -p "../final_output/esrgan_output"
fi

python3 inference_realesrgan.py -n RealESRGAN_x4plus -i ../mask_and_lama_inpainting/output_image -o ../final_output/esrgan_output --face_enhance -s 4

echo 'BSRGAN processing started'
cd ../BSRGAN/
python3 main_bsrgan.py

echo "The entire flow was completed successfully"

# Delete the intermediary inpaint output that is of lower resolution
rm -rf ../mask_and_lama_inpainting/output_image
deactivate
