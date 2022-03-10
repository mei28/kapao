# deepsortの結果を描画する
# !/usr/bin/bash

OUTPUT_DIR='exp/20220228/'

# python exp/20220228/deepsort2kapao.py --weights kapao_s_coco.pt --start 0 --end 3645 --video_name "/media/mei/SSD-PGMU3C/20220228/p008.mp4" --output_dir $OUTPUT_DIR --sorted_mot "exp/20220228/p008_deepsort_output"
python exp/20220228/deepsort2kapao.py --weights kapao_s_coco.pt --start 0 --end 36 --video_name "/media/mei/SSD-PGMU3C/20220228/p008.mp4" --output_dir $OUTPUT_DIR --sorted_mot "exp/20220228/p008_deepsort_output"
