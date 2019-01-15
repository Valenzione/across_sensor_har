#!/bin/bash
# Download http://www.shl-dataset.org/download/ dataset

mkdir SHL

wget -O SHL/part1.zip http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part1.zip
wget -O SHL/part2.zip http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part2.zip
wget -O SHL/part3.zip http://www.shl-dataset.org/wp-content/uploads/SHLDataset_preview_v1_part3.zip

unzip -q SHL/part1.zip -d SHL/
unzip -q SHL/part2.zip -d SHL/
unzip -q SHL/part3.zip -d SHL/

wget -O SHL/data_format.pdf http://www.shl-dataset.org/wp-content/uploads/2017/11/doc_dataset.pdf
