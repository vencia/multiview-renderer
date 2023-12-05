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
        # points = mesh.sample(4000)
        points = mesh.vertices
        projection_matrix = np.load(img_path.parent / f'{sample_id}_projection_matrix.npy')
        inverted_world_matrix = np.load(img_path.parent / f'{sample_id}_inverted_world_matrix.npy')

        pixel_positions, z_depth = points_to_pixels(points, projection_matrix, inverted_world_matrix)
        for i, pixel in enumerate(pixel_positions):
            normed_point = (points[i] + 1) / 2  # normed from [-1,1] to [0,1]
            normed_z_depth = (z_depth[i] - z_depth.min()) / (z_depth.max() - z_depth.min())
            # new_img_array[tuple(pixel)] = (255 * normed_point[0], 255 * normed_point[1], 255 * normed_point[2], 255)
            new_img_array[tuple(pixel)] = (255 * normed_z_depth, 0, 0, 255)
        new_image = transparency_to_white(Image.fromarray(new_img_array))
        new_img_path = projected_img_dir / img_path.relative_to(img_dir)
        os.makedirs(new_img_path.parent, exist_ok=True)
        new_image.save(new_img_path)

        # pixels = np.asarray([[resolution / 2, resolution / 2]])
        # mid_point = pixels_to_points(pixels, projection_matrix, inverted_world_matrix)
        # mesh.vertices = np.concatenate((mesh.vertices, mid_point), 0)
        # mesh.faces = np.concatenate(
        #     (mesh.faces, [(len(mesh.vertices) - 1, len(mesh.vertices) - 2, len(mesh.vertices) - 3)]), 0)
        # new_mesh_path = projected_img_dir / img_path.relative_to(img_dir).with_suffix('.stl')
        # mesh.export(new_mesh_path)


def transparency_to_white(image):
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image.convert("RGB")
    return new_image


def point_to_pixel_matrix(projection_matrix, view_matrix):
    return projection_matrix @ view_matrix


def pixel_to_point_matrix(projection_matrix, view_matrix):
    return np.linalg.inv(projection_matrix @ view_matrix)


def points_to_pixels(points, projection_matrix, view_matrix):
    points = np.concatenate([points, np.ones((len(points), 1))], -1)  # homogenize
    projected = np.asarray([point_to_pixel_matrix(projection_matrix, view_matrix) @ x for x in points])
    projected = projected[:, :-1] / projected[:, -1, None]  # dehomogenize
    pixel_positions, z_depth = projected[:, :2], projected[:, 2]

    # [-1, 1] to [0, 1]
    pixel_positions = (pixel_positions + 1.0) / 2.0

    pixel_positions[:, 1] = 1 - pixel_positions[:, 1]  # flip y axis
    pixel_positions *= resolution  # scale from [0,1] to image size
    pixel_positions = np.stack((pixel_positions[:, 1], pixel_positions[:, 0]), axis=-1)  # switch x and y
    return pixel_positions.astype(int), z_depth


def pixels_to_points(pixel_positions, projection_matrix, view_matrix):
    pixel_positions = np.stack((pixel_positions[:, 1], pixel_positions[:, 0]), axis=-1)  # switch x and y
    pixel_positions /= resolution
    pixel_positions[:, 1] = 1 - pixel_positions[:, 1]  # flip y axis
    projected = pixel_positions * 2.0 - 1.0
    projected = np.concatenate([projected, np.full((len(projected), 1), 0.5)], -1)  # z coordinate (depth)
    projected = np.concatenate([projected, np.ones((len(projected), 1))], -1)  # homogenize
    projected = np.asarray([pixel_to_point_matrix(projection_matrix, view_matrix) @ x for x in projected])
    projected = projected[:, :-1] / projected[:, -1]  # dehomogenize
    return projected


if __name__ == '__main__':
    main()
