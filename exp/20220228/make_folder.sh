#!/usr/bin/bash


array=(`ls . | grep -e "mp4"`)

for file in "${array[@]}"; do
  echo $file

done
