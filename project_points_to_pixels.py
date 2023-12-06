import os
import numpy as np
from PIL import Image
from pathlib import Path
import trimesh
import math
from utils_points import plot_points

resolution = 224
img_dir = Path('/home/vencia/Documents/datasets/dmunet/test_imgs_new')
projected_img_dir = Path('/home/vencia/Documents/datasets/dmunet/test_imgs_new_projected')
mesh_dir = Path('/home/vencia/Documents/datasets/dmunet/STL_dataset_preprocessed')


def main():
    for img_path in sorted(img_dir.rglob('**/*.png')):
        sample_id = img_path.stem
        if '_depth' in sample_id:
            continue
        img = Image.open(img_path).convert("RGBA")
        img_array = np.asarray(img)
        new_img_array = img_array.copy()

        projection_matrix = np.load(img_path.parent / f'{sample_id}_projection_matrix.npy')
        view_matrix = np.load(img_path.parent / f'{sample_id}_view_matrix.npy')
        depth_values = np.load(img_path.parent / f'{sample_id}_depth.npy')
        # projected_pc = np.load(img_path.parent / f'{sample_id}_projected_pointcloud.npy')
        camera_data = np.load(img_path.parent / f'{sample_id}_camera_data.npz')

        projected_pc = point_cloud(depth_values, camera_data)
        projected_pc = projected_pc.reshape(-1, 3)
        projected_pc = projected_pc[np.isnan(projected_pc[:, -1]) == False]

        projected_pc_path = projected_img_dir / img_path.relative_to(img_dir).with_suffix('.vtk')
        os.makedirs(projected_pc_path.parent, exist_ok=True)
        plot_points(projected_pc_path, projected_pc)

        # non_background_depth_values = depth_values[depth_values < 1000]
        # non_background_z_values = z_values[z_values < 1000]
        # print('exported depth range', non_background_depth_values.min(), 'to', non_background_depth_values.max())
        # print('exported z range', non_background_z_values.min(), 'to', non_background_z_values.max())

        # mesh = trimesh.load(mesh_dir / img_path.parent.parent.stem / f'{img_path.parent.stem}.obj')
        #
        # # pixel_point_map = np.full((resolution, resolution, 4), np.nan)
        # points = mesh.sample(1000)
        # pixel_positions, z_depth = points_to_pixels(points, projection_matrix, view_matrix)
        # print('projected depth range', z_depth.min(), 'to', z_depth.max())

        # for i, pixel in enumerate(pixel_positions):
        #     if np.isnan(pixel_point_map[tuple(pixel)][-1]) or z_depth[i] < pixel_point_map[tuple(pixel)][-1]:
        #         pixel_point_map[tuple(pixel)] = (points[i][0], points[i][1], points[i][2], z_depth[i])
        #
        # projected_pc_path = projected_img_dir / img_path.relative_to(img_dir).with_suffix('.vtk')
        # os.makedirs(projected_pc_path.parent, exist_ok=True)
        # pc = pixel_point_map.reshape(resolution * resolution, -1)
        # pc = pc[np.isnan(pc)[:, 0] == False]
        # plot_points(projected_pc_path, pc[:, :3], scalar_fields={'z_depth': pc[:, -1]})

        # points = mesh.vertices

        # pixel_positions, z_depth = points_to_pixels(points, projection_matrix, view_matrix)
        # for i, pixel in enumerate(pixel_positions):
        #     normed_point = (points[i] + 1) / 2  # normed from [-1,1] to [0,1]
        #     normed_z_depth = (z_depth[i] - z_depth.min()) / (z_depth.max() - z_depth.min())
        #     # new_img_array[tuple(pixel)] = (255 * normed_point[0], 255 * normed_point[1], 255 * normed_point[2], 255)
        #     new_img_array[tuple(pixel)] = (255 * normed_z_depth, 0, 0, 255)
        # new_image = transparency_to_white(Image.fromarray(new_img_array))
        # new_img_path = projected_img_dir / img_path.relative_to(img_dir)
        # os.makedirs(new_img_path.parent, exist_ok=True)
        # new_image.save(new_img_path)

        # pixels = np.asarray([[resolution / 2, resolution / 2]]).astype(int)
        # depths = np.asarray([depth_values[tuple(x)] for x in pixels])
        # mid_point = pixels_to_points(pixels, depths, projection_matrix, view_matrix)
        # mesh.vertices = np.concatenate((mesh.vertices, mid_point), 0)
        # mesh.faces = np.concatenate(
        #     (mesh.faces, [(len(mesh.vertices) - 1, len(mesh.vertices) - 2, len(mesh.vertices) - 3)]), 0)
        # new_mesh_path = projected_img_dir / img_path.relative_to(img_dir).with_suffix('.stl')
        # os.makedirs(new_mesh_path.parent, exist_ok=True)
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
    pixel_positions = (pixel_positions + 1.0) / 2.0  # [-1, 1] to [0, 1]
    pixel_positions[:, 1] = 1 - pixel_positions[:, 1]  # flip y axis
    pixel_positions *= resolution  # scale from [0,1] to image size
    pixel_positions = np.stack((pixel_positions[:, 1], pixel_positions[:, 0]), axis=-1)  # switch x and y
    return pixel_positions.astype(int), z_depth


def pixels_to_points(pixel_positions, depths, projection_matrix, view_matrix):
    pixel_positions = pixel_positions.astype(float)
    pixel_positions = np.stack((pixel_positions[:, 1], pixel_positions[:, 0]), axis=-1)  # switch x and y
    pixel_positions /= resolution
    pixel_positions[:, 1] = 1 - pixel_positions[:, 1]  # flip y axis
    projected = pixel_positions * 2.0 - 1.0  # [0, 1] to [-1, 1]
    projected = np.concatenate([projected, depths[:, None]], -1)  # z coordinate (depth)
    projected = np.concatenate([projected, np.ones((len(projected), 1))], -1)  # homogenize
    projected = np.asarray([pixel_to_point_matrix(projection_matrix, view_matrix) @ x for x in projected])
    projected = projected[:, :-1] / projected[:, -1]  # dehomogenize
    return projected


def point_cloud(depth, camera_data):
    # Distance factor from the cameral focal angle
    factor = 2.0 * math.tan(camera_data['angle_x'] / 2.0)

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    # Valid depths are defined by the camera clipping planes
    valid = (depth > camera_data['clip_start']) & (depth < camera_data['clip_end'])

    # Negate Z (the camera Z is at the opposite)
    z = -np.where(valid, depth, np.nan)

    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows, cols)
    x = -np.where(valid, factor * z * (c - (cols / 2)) / ratio, 0)
    y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, 0)

    return np.dstack((x, y, z))


if __name__ == '__main__':
    main()
