#!/bin/bash

source env_m_i_u/bin/activate

cd ./Real-ESRGAN/

 for i in {1..4}; do
     python3 inference_realesrgan.py -n RealESRGAN_x4plus -i ../initial_input/upscaler_input_dir -o ../final_output/esrgan_output --face_enhance -s $i
 done

echo "The upscale process completed successfully"
deactivate
