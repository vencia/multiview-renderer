from PIL import Image
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render_dir', type=str, default='data/datasets/dmunet/STL_dataset_test_imgs')
parser.add_argument('--output_dir', type=str, default='data/datasets/dmunet/STL_dataset_imgs_white')
args = parser.parse_args()

render_dir = Path(args.render_dir)
output_dir = Path(args.output_dir)

for image_path in sorted(render_dir.rglob('**/*.png')):
    sample_id = image_path.stem
    output_folder = output_dir / image_path.parent.relative_to(render_dir)
    os.makedirs(output_folder, exist_ok=True)

    image = Image.open(image_path).convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image.convert("RGB").save(output_folder / f'{sample_id}.png')
