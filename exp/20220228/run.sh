# !/usr/bin/bash

OUTPUT_DIR='exp/20220228/'

# python exp/20220228/main.py --weights kapao_s_coco.pt --start 0 --end 3645 --video_name "/media/mei/SSD-PGMU3C/20220228/p008.mp4" --output_dir $OUTPUT_DIR
# python exp/20220228/main.py --weights kapao_s_coco.pt --start 0 --end 1260 --video_name "/media/mei/SSD-PGMU3C/20220228/p029.mp4" --output_dir $OUTPUT_DIR 
python exp/20220228/main.py --weights kapao_s_coco.pt --start 0 --end 2010 --video_name "/media/mei/SSD-PGMU3C/20220228/p036.mp4" --output_dir $OUTPUT_DIR
python exp/20220228/main.py --weights kapao_s_coco.pt --start 0 --end 4080 --video_name "/media/mei/SSD-PGMU3C/20220228/p037.mp4" --output_dir $OUTPUT_DIR
python exp/20220228/main.py --weights kapao_s_coco.pt --start 0 --end 2550 --video_name "/media/mei/SSD-PGMU3C/20220228/p042.mp4" --output_dir $OUTPUT_DIR


