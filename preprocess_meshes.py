import glob
import os
import trimesh
import numpy as np
from pathlib import Path
import meshio

mesh_dir = Path('data/datasets/dmunet/STL_dataset')
preprocessed_dir = Path('data/datasets/dmunet/STL_dataset_preprocessed')


def main():
    for mesh_path in sorted(mesh_dir.glob('**/*.stl'))[:10]:
        sample_id = mesh_path.stem
        output_dir = preprocessed_dir / mesh_path.parent.relative_to(mesh_dir)
        os.makedirs(str(output_dir), exist_ok=True)
        output_path = str(output_dir / (sample_id + '.obj'))
        if os.path.exists(output_path):
            continue

        print(sample_id, '...')
        mesh = trimesh.load(mesh_path)
        # mesh = load_from_vtk(sample_path)
        scale = (mesh.bounds[1] - mesh.bounds[0]).max()
        center = (mesh.bounds[1] + mesh.bounds[0]) / 2
        mesh.vertices = (mesh.vertices - center[None, :]) / scale  # normalize
        mesh.vertices = np.stack((-mesh.vertices[:, 0], mesh.vertices[:, 2], mesh.vertices[:, 1]),
                                 -1)  # change y and z axis, flip x axis

        mesh.export(output_path)


def load_from_vtk(vtk_path):
    vtk_mesh = meshio.read(vtk_path)
    mesh = trimesh.Trimesh(vertices=vtk_mesh.points, faces=vtk_mesh.cells[0].data,
                           face_attributes={k: v[0] for k, v in vtk_mesh.cell_data.items()})
    return mesh


if __name__ == '__main__':
    main()
