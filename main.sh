# !/usr/bin/bash

# 学習済みパラメータ・ダウンロード
# 一度だけ
# ! sh data/scripts/download_models.sh

VIDEO_PATH="./movie/mp4/"
VIDEO_NAME='p002.mp4'

python test.py --name $VIDEO_PATH$VIDEO_NAME --weights kapao_s_coco.pt --start 0 --end 1200

mv output.mp4 "./movie/pose-$VIDEO_NAME"
# xdg-open output.mp4
