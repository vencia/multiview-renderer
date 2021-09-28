import glob
import os
import trimesh
import numpy as np
from pathlib import Path

from settings import g_mesh_dataset_path, g_preprocessed_dataset_path


def main():
    for sample_path in glob.glob(g_mesh_dataset_path + '/**/*.off', recursive=True):
        sample_id = Path(sample_path).stem
        print(sample_id, '...')
        mesh = trimesh.load(sample_path)
        scale = (mesh.bounds[1] - mesh.bounds[0]).max()
        center = (mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.vertices = (mesh.vertices - center[None, :]) / scale  # normalize
        mesh.vertices = np.stack((-mesh.vertices[:, 0], mesh.vertices[:, 2], mesh.vertices[:, 1]),
                                 -1)  # change y and z axis, flip x axis
        output_dir = g_preprocessed_dataset_path / Path(sample_path).parent.relative_to(g_mesh_dataset_path)
        os.makedirs(str(output_dir), exist_ok=True)
        mesh.export(str(output_dir / (sample_id + '.obj')))


if __name__ == '__main__':
    main()
