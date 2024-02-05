import os
import numpy as np
from PIL import Image
from pathlib import Path
import trimesh
import math
from utils_points import plot_points
import argparse

parser = argparse.ArgumentParser(
    description='Compute correspondence between 3d points and 2d pixels in both directions.')
parser.add_argument('--render_dir', type=str, default='data/datasets/dmunet/STL_dataset_imgs')
parser.add_argument('--correspondence_dir', type=str, default='data/datasets/dmunet/STL_dataset_imgs_correspondence_test')
parser.add_argument('--pointcloud_dir', type=str, default='data/datasets/dmunet/points_with_normals')
parser.add_argument('--overwrite', type=bool, default=True)
parser.add_argument('--resolution', type=int, default=224)
args = parser.parse_args()

img_dir = Path(args.render_dir)
correspondence_dir = Path(args.correspondence_dir)
pc_dir = Path(args.pointcloud_dir)


def main():
    for img_path in sorted(img_dir.rglob('**/*.png')):
        sample_id = img_path.stem
        projected_img_path = correspondence_dir / img_path.relative_to(img_dir)
        projected_pc_path = correspondence_dir / img_path.relative_to(img_dir).with_suffix('.vtk')

        if '_depth' in sample_id:
            continue
        if not args.overwrite and os.path.isfile(projected_img_path) and os.path.isfile(projected_pc_path):
            print(f'{sample_id} already exists, skip.')
            continue

        img = Image.open(img_path).convert("RGBA")
        new_img = img.copy()
        projection_matrix = np.load(img_path.parent / f'{sample_id}_projection_matrix.npy')
        view_matrix = np.load(img_path.parent / f'{sample_id}_view_matrix.npy')
        points = np.load(pc_dir / img_path.parent.parent.stem / f'{img_path.parent.stem}.npy')[:, :3]
        points = normalize(points)
        # points = np.stack((-points[:, 0], points[:, 2], points[:, 1]), -1)
        overlay_img = pointcloud_to_image(points, projection_matrix, view_matrix)
        new_img.paste(overlay_img, (0, 0), overlay_img)
        os.makedirs(projected_img_path.parent, exist_ok=True)
        new_img.save(projected_img_path)

        depth = np.load(img_path.parent / f'{sample_id}_depth.npy')
        camera_data = np.load(img_path.parent / f'{sample_id}_camera_data.npz')
        points, pixel_pos = image_to_pointcloud(depth, camera_data)
        os.makedirs(projected_pc_path.parent, exist_ok=True)
        plot_points(projected_pc_path, points, vector_fields={'pixel_pos':pixel_pos})


def normalize(points):
    scale = (points.max(0) - points.min(0)).max()
    center = (points.max(0) + points.min(0)) / 2.0
    points = (points - center[None, :]) / scale
    return points


def transparency_to_white(image):
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    new_image.convert("RGB")
    return new_image


def image_to_pointcloud(depth, camera_data):
    pc = point_cloud(depth, camera_data)
    pixel_pos = np.indices((args.resolution,args.resolution)).swapaxes(0,1).swapaxes(1,2)
    pc = pc.reshape(args.resolution * args.resolution,-1)
    pixel_pos = pixel_pos.reshape(args.resolution * args.resolution,-1)
    mask = np.isnan(pc[:, 0]) == False
    pc = pc[mask]
    pixel_pos = pixel_pos[mask]
    return pc, pixel_pos


def pointcloud_to_image(points, projection_matrix, view_matrix):
    img_array = np.zeros((args.resolution, args.resolution, 4), dtype=np.uint8)
    pixel_positions, z_depth = points_to_pixels(points, projection_matrix, view_matrix)
    for i, pixel in enumerate(pixel_positions):
        # normed_point = (points[i] + 1) / 2  # normed from [-1,1] to [0,1]
        # img_array[tuple(pixel)] = (255 * normed_point[0], 255 * normed_point[1], 255 * normed_point[2], 255)
        normed_z_depth = (z_depth[i] - z_depth.min()) / (z_depth.max() - z_depth.min())
        if (0 <= pixel).all() and (pixel < args.resolution).all():
            img_array[tuple(pixel)] = (255 * normed_z_depth, 0, 0, 255)
        else:
            print(f'warning, pixel {pixel} not in range [0,{args.resolution - 1}].')
    return Image.fromarray(img_array)


def points_to_pixels(points, projection_matrix, view_matrix):
    points = np.concatenate([points, np.ones((len(points), 1))], -1)  # homogenize
    projected = np.asarray([projection_matrix @ view_matrix @ x for x in points])
    projected = projected[:, :-1] / projected[:, -1, None]  # dehomogenize
    pixel_positions, z_depth = projected[:, :2], projected[:, 2]
    pixel_positions = (pixel_positions + 1.0) / 2.0  # [-1, 1] to [0, 1]
    pixel_positions[:, 1] = 1 - pixel_positions[:, 1]  # flip y axis
    pixel_positions *= args.resolution  # scale from [0,1] to image size
    pixel_positions = np.stack((pixel_positions[:, 1], pixel_positions[:, 0]), axis=-1)  # switch x and y
    return pixel_positions.astype(int), z_depth


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
    x = np.where(valid, factor * z * (c - (cols / 2)) / ratio, np.nan)
    y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, np.nan)

    return np.dstack((-x, -y, z))


if __name__ == '__main__':
    main()
