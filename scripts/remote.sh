#!/bin/bash

echo "copy dataset .."

src_dir="/mnt/no_auth_team/render_shapenet/huggingface"
dest_dir="/src/data/datasets/shapenet/huggingface"

folder="04468005"

mkdir -p $dest_dir

rsync $src_dir/shapenetcore_taxonomy.json $dest_dir/
rsync $src_dir/$folder.zip $dest_dir/
#unzip -q $dest_dir/02773838.zip -d $dest_dir/


echo "render remote ..."
python render_shapenet_batch.py --overwrite --engine CYCLES --mesh_dir $dest_dir --render_dir /mnt/no_auth_team/render_shapenet/huggingface_imgs --blender_executable blender
