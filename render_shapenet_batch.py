import os.path
from pathlib import Path
import zipfile
import subprocess
import tqdm
import shutil
import json
import argparse

# NO BACKGROUND RENDERING FOR GETTING ACTUAL DEPTH VALUES!


parser = argparse.ArgumentParser(description='Renders given folder of stl/obj meshes.')
parser.add_argument('--mesh_dir', type=str, default='data/datasets/shapenet/huggingface')
parser.add_argument('--render_dir', type=str, default='data/datasets/shapenet/huggingface_imgs')
parser.add_argument('--overwrite', dest='overwrite', action='store_true')
parser.add_argument('--blender_executable', type=str, default='/home/vencia/blender-2.93.0/blender')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE')
args = parser.parse_args()

mesh_dir = Path(args.mesh_dir)
render_dir = Path(args.render_dir)
taxonomy_path = mesh_dir / 'shapenetcore_taxonomy.json'

ignore = ['03001627/c5c4e6110fbbf5d3d83578ca09f86027']

with open(taxonomy_path, 'r') as f:
    taxonomy = json.load(f)
    num_samples_per_category = {x['metadata']['name']: x['metadata']['numInstances'] for x in taxonomy}
    num_samples_per_category['03001627'] -= 1  # because of ignored sample
    num_samples_per_category['02992529'] = 831  # category is not in taxonomy

for zip_file in sorted(mesh_dir.glob('*.zip')):
    category = zip_file.stem
    unzipped_folder = zip_file.parent / category

    if category not in num_samples_per_category:
        print(f'warning: {category} not in taxonomy')
    elif not args.overwrite and len(list((render_dir / category).rglob('**/*.png'))) == 20 * num_samples_per_category[
        category]:
        assert len(list((render_dir / category).rglob('**/*.npy'))) == 40 * num_samples_per_category[category]
        print(f'skip {category}, already completely rendered.')
        continue

    if os.path.isdir(unzipped_folder):
        print(f'skip unzipped folder {unzipped_folder.stem}, already exists.')
        continue
    else:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(zip_file.parent)

    print(f'render {category} ...')

    for sample_path in tqdm.tqdm(unzipped_folder.iterdir()):
        assert sample_path.is_dir()
        mesh_path = sample_path / 'models' / 'model_normalized.obj'
        assert os.path.isfile(mesh_path)
        render_path = render_dir / category / sample_path.stem / 'models' / 'model_normalized'

        if not args.overwrite and os.path.isfile(render_path / f'{render_path.stem}_020.png'):
            # print(f'skip {sample_path.stem}, already rendered.')
            continue

        if f'{category}/{sample_path.stem}' in ignore:
            print(f'ignore {sample_path.stem}, skip.')
            continue

        # print(f'render {sample_path.stem} ...')

        command = [
            args.blender_executable,
            "--background",
            "--python",
            "render_mesh.py",
            "--",
            "--mesh_path",
            str(mesh_path),
            "--render_path",
            str(render_path),
            "--engine",
            args.engine
        ]

        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    shutil.rmtree(unzipped_folder)
