import glob
import os
import trimesh
import numpy as np
from pathlib import Path
import meshio
from settings import g_mesh_dataset_path, g_preprocessed_dataset_path


def main():
    for sample_path in sorted(glob.glob(g_mesh_dataset_path + '/**/*.stl', recursive=True)):
        sample_id = Path(sample_path).stem
        output_dir = g_preprocessed_dataset_path / Path(sample_path).parent.relative_to(g_mesh_dataset_path)
        os.makedirs(str(output_dir), exist_ok=True)
        output_path = str(output_dir / (sample_id + '.obj'))
        if os.path.exists(output_path):
            continue

        print(sample_id, '...')
        mesh = trimesh.load(sample_path)
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
