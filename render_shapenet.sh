#!/bin/bash

# NO BACKGROUND RENDERING FOR GETTING ACTUAL DEPTH VALUES!

data_path="/home/vencia/Documents/datasets/shapenet/huggingface"
render_path="/home/vencia/Documents/datasets/shapenet/huggingface_imgs"

for zip_file in "$data_path"/*.zip; do
  if [ -f "$zip_file" ]; then
    # Extract the base name of the zip file
    zip_file_name=$(basename "$zip_file")

        # Check if the unzipped folder already exists
    if [ -d "$data_path/${zip_file_name%.*}" ]; then
      echo "Unzipped folder already exists. Skipping: $zip_file_name"
      continue
    fi

    echo "$zip_file_name"

#    # Unzip the zip file into the unzip directory
    unzip -o "$zip_file" -d "$data_path" >/dev/null 2>&1
    mesh_dir=$data_path/${zip_file_name%.*}
    echo "mesh_dir: $mesh_dir"
    render_dir=$render_path/${zip_file_name%.*}
    echo "render_dir: $render_dir"

    /home/vencia/blender-2.93.0/blender --background --python render.py -- --mesh_dir "$mesh_dir" --render_dir "$render_dir" >/dev/null 2>&1

    rm -rf "$mesh_dir"

  fi
done

