import bpy
import os
import numpy as np
from pathlib import Path
import math
import argparse
import sys

parser = argparse.ArgumentParser(description='Renders given folder of obj meshes.')
parser.add_argument('--mesh_dir', type=str, default='data/datasets/dmunet/STL_dataset_preprocessed')
parser.add_argument('--render_dir', type=str, default='data/datasets/dmunet/test_imgs_new')
parser.add_argument('--num_views', type=int, default=20, choices=[12, 20],
                    help='number of views to be rendered')
parser.add_argument('--overwrite', type=bool, default=True)
parser.add_argument('--fit_view', type=bool, default=False)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=224 * 2)  # 224
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

scene = bpy.context.scene
camera = scene.camera


def main():
    init_all()

    mesh_dir = Path(args.mesh_dir)
    render_dir = Path(args.render_dir)

    for sample_path in sorted(mesh_dir.rglob('**/*.obj')):
        sample_id = sample_path.stem
        output_folder = render_dir / sample_path.parent.relative_to(mesh_dir) / sample_id
        if not args.overwrite and os.path.isdir(output_folder) and len(
                [x for x in os.listdir(output_folder) if x.endswith('.png')]) == args.num_views:
            print(f'{sample_id} already exists, skip.')
            continue

        print(sample_id, '...')
        os.makedirs(output_folder, exist_ok=True)

        clear_mesh()

        bpy.ops.import_scene.obj(filepath=str(sample_path))

        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

        if args.scale != 1:
            bpy.ops.transform.resize(value=(args.scale, args.scale, args.scale))
            bpy.ops.object.transform_apply(scale=True)

        # Set objekt IDs
        obj.pass_index = 1

        circumradius = math.sqrt(3)
        distance = circumradius * 0.8
        tilt = 0.0
        if args.num_views == 12:  # elevated circle of cameras
            azimuths = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            elevations = np.full(12, fill_value=np.pi / 4)
        elif args.num_views == 20:
            azimuths, elevations = _dodecahedron(circumradius)
        else:
            assert False

        for view_id in range(args.num_views):
            azimuth = azimuths[view_id]
            elevation = elevations[view_id]
            cam_loc = camera_location(azimuth, elevation, distance)
            cam_rot = camera_rot_XYZEuler(azimuth, elevation, tilt)
            scene.frame_set(view_id + 1)
            scene.render.filepath = str(output_folder / f'{sample_id}_{view_id:03d}.png')
            render_with_cam(cam_loc, cam_rot, obj, output_folder / f'{sample_id}_{view_id:03d}')


def render_with_cam(cam_loc, cam_rot, obj, output_path):
    # cam_obj = scene.objects['Camera']
    camera.location = cam_loc
    camera.rotation_euler = cam_rot
    camera.data.lens = 35
    camera.data.sensor_width = 32
    #
    # cam_constraint = cam_obj.constraints.new(type='TRACK_TO')
    # cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    # cam_constraint.up_axis = 'UP_Y'

    if args.fit_view:
        bpy.ops.view3d.camera_to_view_selected()

    assert scene.render.resolution_percentage == 100

    projection_matrix = get_projection_matrix()
    inverted_world_matrix = get_inverted_world_matrix()
    np.save(f'{output_path}_projection_matrix.npy', projection_matrix)
    np.save(f'{output_path}_inverted_world_matrix.npy', inverted_world_matrix)

    # coords_2d = project_with_matrix(obj)
    # coords_2d[:, 1] = 1 - coords_2d[:, 1]  # flip y axis
    # coords_2d *= res  # scale from [0,1] to image size
    # coords_2d = np.stack((coords_2d[:, 1], coords_2d[:, 0]), axis=-1)  # switch x and y
    # np.save(f'{output_path}_vertices_to_pixels.npy', coords_2d)

    bpy.ops.render.render(write_still=True)


def clear_mesh():
    """ clear all meshes in the scene

    """
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' or obj.type == 'EMPTY':
            obj.select_set(state=True)
    bpy.ops.object.delete()


def init_all():
    scene.render.engine = args.engine
    scene.render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
    scene.render.image_settings.color_depth = args.color_depth  # ('8', '16')
    scene.render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True

    scene.use_nodes = True
    # scene.view_layers["View Layer"].use_pass_normal = True
    # scene.view_layers["View Layer"].use_pass_diffuse_color = True
    # scene.view_layers["View Layer"].use_pass_object_index = True

    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    # Delete default cube
    bpy.context.active_object.select_set(True)
    bpy.ops.object.delete()

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 0.0
    light.energy = 5.0

    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 0.0
    light2.energy = 5.0  # 0.015
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] = 180


def camera_location(azimuth, elevation, dist):
    """get camera_location (x, y, z)

    you can write your own version of camera_location function
    to return the camera loation in the blender world coordinates
    system

    Args:
        azimuth: azimuth radius(object centered)
        elevation: elevation radius(object centered)
        dist: distance between camera and object(in meter)

    Returens:
        return the camera location in world coordinates in meters
    """

    phi = float(elevation)
    theta = float(azimuth)
    dist = float(dist)

    x = dist * math.cos(phi) * math.cos(theta)
    y = dist * math.cos(phi) * math.sin(theta)
    z = dist * math.sin(phi)

    return x, y, z


def camera_rot_XYZEuler(azimuth, elevation, tilt):
    """get camera rotaion in XYZEuler

    Args:
        azimuth: azimuth radius(object centerd)
        elevation: elevation radius(object centerd)
        tilt: twist radius(object centerd)

    Returns:
        return the camera rotation in Euler angles(XYZ ordered) in radians
    """

    azimuth, elevation, tilt = float(azimuth), float(elevation), float(tilt)
    x, y, z = math.pi / 2, 0, math.pi / 2  # set camera at x axis facing towards object

    # latitude
    x = x - elevation
    # longtitude
    z = z + azimuth

    return x, y, z


def _dodecahedron(circumradius):
    # https://github.com/yinyunie/depth_renderer
    phi = (1 + math.sqrt(5)) / 2.  # golden_ratio
    dodecahedron = [[-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [0, -phi, -1 / phi],
                    [0, -phi, 1 / phi],
                    [0, phi, -1 / phi],
                    [0, phi, 1 / phi],
                    [-1 / phi, 0, -phi],
                    [-1 / phi, 0, phi],
                    [1 / phi, 0, -phi],
                    [1 / phi, 0, phi],
                    [-phi, -1 / phi, 0],
                    [-phi, 1 / phi, 0],
                    [phi, -1 / phi, 0],
                    [phi, 1 / phi, 0]]
    elevations = [math.asin(x[2] / circumradius) for x in dodecahedron]
    azimuths = [math.atan2(x[1], x[0]) for x in dodecahedron]
    return azimuths, elevations


def get_projection_matrix():
    width, height = scene.render.resolution_x, scene.render.resolution_y
    projection_matrix = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)
    return np.asarray(projection_matrix)


def get_inverted_world_matrix():
    inverted_world_matrix = camera.matrix_world.inverted()
    return np.asarray(inverted_world_matrix)


def project_with_matrix(obj):
    width, height = scene.render.resolution_x, scene.render.resolution_y
    projection_matrix = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=width, y=height)
    projection_matrix = np.array([list(row) for row in projection_matrix])

    # vertices in camera space
    verts_camspace = np.array([list(camera.matrix_world.inverted() @ v.co) for v in obj.data.vertices])

    # Homogenize
    verts_camspace_h = np.hstack([verts_camspace, np.ones((len(verts_camspace), 1))])

    # Project
    projected = verts_camspace_h.dot(projection_matrix.T)

    # Dehomogenize
    projected = projected[:, :2] / projected[:, 3, None]

    # [-1, 1] to [0, 1]
    projected = (projected + 1.0) / 2.0

    return projected


if __name__ == '__main__':
    main()
