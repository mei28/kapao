# !/usr/bin/bash

OUTPUT_DIR='exp/20231102/'

python exp/20231102/main.py --weights kapao_s_coco.pt --start 0 --end 3645 --video_name "~/Documents/predict_pingpong/outputs/inference/exp25_multimlp_0.mp4" --output_dir $OUTPUT_DIR
