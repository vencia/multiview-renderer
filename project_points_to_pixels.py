import os

import numpy as np
from PIL import Image
from pathlib import Path
import trimesh

resolution = 224 * 2
img_dir = Path('/home/vencia/Documents/datasets/dmunet/test_imgs_new')
projected_img_dir = Path('/home/vencia/Documents/datasets/dmunet/test_imgs_new_projected')
mesh_dir = Path('/home/vencia/Documents/datasets/dmunet/STL_dataset_preprocessed')


def main():
    for img_path in sorted(img_dir.rglob('**/*.png')):
        sample_id = img_path.stem
        img = Image.open(img_path).convert("RGBA")
        img_array = np.asarray(img)
        new_img_array = img_array.copy()

        mesh = trimesh.load(mesh_dir / img_path.parent.parent.stem / f'{img_path.parent.stem}.obj')
        points = mesh.sample(4000)
        projection_matrix = np.load(img_path.parent / f'{sample_id}_projection_matrix.npy')
        inverted_world_matrix = np.load(img_path.parent / f'{sample_id}_inverted_world_matrix.npy')

        pixel_positions = project_with_matrix(points, projection_matrix, inverted_world_matrix)
        pixel_positions[:, 1] = 1 - pixel_positions[:, 1]  # flip y axis
        pixel_positions *= resolution  # scale from [0,1] to image size
        pixel_positions = np.stack((pixel_positions[:, 1], pixel_positions[:, 0]), axis=-1)  # switch x and y
        pixel_positions = pixel_positions.astype(int)

        # vertices_to_pixels_path = img_path.parent / f'{sample_id}_vertices_to_pixels.npy'
        # pixel_pos = np.load(vertices_to_pixels_path).astype(int)
        for point, pixel in zip(points, pixel_positions):
            normed_point = (point + 1) / 2  # normed from [-1,1] to [0,1]
            new_img_array[tuple(pixel)] = (255 * normed_point[0], 255 * normed_point[1], 255 * normed_point[2], 255)
        new_image = Image.fromarray(new_img_array)
        new_img_path = projected_img_dir / img_path.relative_to(img_dir)
        os.makedirs(new_img_path.parent, exist_ok=True)
        new_image.save(new_img_path)


def project_with_matrix(points, projection_matrix, inverted_world_matrix):
    # Homogenize
    points = np.concatenate([points, np.ones((len(points), 1))], -1)

    # vertices in camera space
    verts_camspace = np.array([list(inverted_world_matrix @ x) for x in points])

    # Project
    projected = verts_camspace.dot(projection_matrix.T)

    # Dehomogenize
    projected = projected[:, :2] / projected[:, 3, None]

    # [-1, 1] to [0, 1]
    projected = (projected + 1.0) / 2.0

    return projected


if __name__ == '__main__':
    main()
