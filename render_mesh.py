import time

import bpy
from mathutils import Vector
import os
import numpy as np
from pathlib import Path
import math
import argparse
import sys

parser = argparse.ArgumentParser(description='Renders given folder of stl/obj meshes.')
parser.add_argument('--mesh_path', type=str,
                    default='data/datasets/shapenet/huggingface_test/02843684/1b73f96cf598ef492cba66dc6aeabcd4/models/model_normalized.obj')
parser.add_argument('--mesh_format', type=str, default='obj')  # stl
parser.add_argument('--render_path', type=str,
                    default='data/datasets/shapenet/huggingface_test_imgs/02843684/1b73f96cf598ef492cba66dc6aeabcd4/models/model_normalized')
parser.add_argument('--num_views', type=int, default=20, choices=[12, 20],
                    help='number of views to be rendered')
# parser.add_argument('--overwrite', type=bool, default=False)
# parser.add_argument('--fit_view', type=bool, default=False)
# parser.add_argument('--scale', type=float, default=0.4)
parser.add_argument('--normalize', type=bool, default=True, help='Normalize object dimensions to range [-0.5,0.5]')
parser.add_argument('--render_depth', type=bool, default=False)
parser.add_argument('--depth_scale', type=float, default=0.9,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=224)
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

bpy.context.preferences.addons['cycles'].preferences.get_devices()
print('gpus found:', len(bpy.context.preferences.addons['cycles'].preferences.devices))

scene = bpy.context.scene
camera = scene.camera


def main():
    sample_path = Path(args.mesh_path)
    output_folder = Path(args.render_path)

    sample_id = sample_path.stem
    print(sample_id, '...')
    os.makedirs(output_folder, exist_ok=True)

    init_all()
    clear_mesh()

    if args.mesh_format == 'obj':
        bpy.ops.import_scene.obj(filepath=str(sample_path))
    else:
        assert args.mesh_format == 'stl'
        bpy.ops.import_mesh.stl(filepath=str(sample_path))

    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.transform_apply()  # need to apply potential scene import rotations

    if args.normalize:
        bbox = np.asarray(obj.bound_box)
        scale = (bbox.max(0) - bbox.min(0)).max()
        center = (bbox.max(0) + bbox.min(0)) / 2.0
        bpy.ops.transform.translate(value=(-center[0], -center[1], -center[2]))
        bpy.ops.object.transform_apply()
        bpy.ops.transform.resize(value=(1 / scale, 1 / scale, 1 / scale))
        bpy.ops.object.transform_apply()

    print('bbox after normalization', np.asarray(obj.bound_box))
    print('dimensions after normalization', np.asarray(obj.dimensions))

    # Set objekt IDs
    obj.pass_index = 1

    circumradius = math.sqrt(3)
    distance = circumradius * 1.0
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
        render_path = str(output_folder / f'{sample_id}_{view_id + 1 :03d}')
        scene.render.filepath = render_path
        if args.render_depth:
            scene.node_tree.nodes['File Output'].file_slots[0].path = str(output_folder / f'{sample_id}_###_depth')

        render_with_cam(cam_loc, cam_rot)

        if args.render_depth:
            # get viewer pixels
            depth_values = bpy.data.images['Viewer Node'].pixels
            depth_values = np.copy(np.array(depth_values))
            depth_values = depth_values.reshape(args.resolution, args.resolution, -1)[:, :, 0]
            np.save(render_path + '_depth.npy', depth_values)

        np.savez(render_path + '_camera_data.npz', **get_camera_data())
        np.save(render_path + '_projection_matrix.npy', get_projection_matrix())
        np.save(render_path + '_view_matrix.npy', get_view_matrix())

        # # Calculate the points
        # points = point_cloud(depth_values)
        # # np.save(output_folder / f'{sample_id}_{view_id + 1:03d}_point_cloud.npy', points)
        #
        # # ----------- show point cloud in scene -------------
        # # Translate the points
        # verts = [camera.matrix_world @ Vector(p) for r in points for p in r]
        # # Create a mesh from the points
        # mesh_data = bpy.data.meshes.new("result")
        # mesh_data.from_pydata(verts, [], [])
        # mesh_data.update()
        # # Create an object with this mesh
        # obj = bpy.data.objects.new("result", mesh_data)
        # scene.collection.objects.link(obj)


def render_with_cam(cam_loc, cam_rot):
    camera.location = cam_loc
    camera.rotation_euler = cam_rot
    camera.data.lens = 35
    camera.data.sensor_width = 32

    cam_constraint = camera.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'

    # if args.fit_view:
    #     bpy.ops.view3d.camera_to_view_selected()

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
    # scene.render.image_settings.compression = 15

    scene.render.use_compositing = True
    scene.use_nodes = True
    scene.view_layers["View Layer"].use_pass_z = True
    scene.view_layers[0].use_pass_z = True

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

    if args.render_depth:

        # Create depth output nodes
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = args.format
        depth_file_output.format.color_depth = args.color_depth
        if args.format == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        else:
            depth_file_output.format.color_mode = "BW"

            # Remap as other types can not represent the full range of depth.
            map = nodes.new(type="CompositorNodeMapValue")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.offset = [-1.0]
            map.size = [args.depth_scale]
            map.use_min = True
            map.min = [0]

            links.new(render_layers.outputs['Depth'], map.inputs[0])
            links.new(map.outputs[0], depth_file_output.inputs[0])

        # create output node
        v = nodes.new('CompositorNodeViewer')
        v.use_alpha = False
        # links.new(render_layers.outputs[0], v.inputs[0])  # link Image to Viewer Image RGB
        links.new(render_layers.outputs['Depth'], v.inputs[0])  # link Z to output

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


def get_view_matrix():
    view_matrix = camera.matrix_world.inverted()
    return np.asarray(view_matrix)


def get_camera_data():
    return {'clip_start': camera.data.clip_start,
            'clip_end': camera.data.clip_end,
            'angle_x': camera.data.angle_x}


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


def point_cloud(depth):
    # Distance factor from the cameral focal angle
    factor = 2.0 * math.tan(camera.data.angle_x / 2.0)

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    # Valid depths are defined by the camera clipping planes
    valid = (depth > camera.data.clip_start) & (depth < camera.data.clip_end)

    # Negate Z (the camera Z is at the opposite)
    z = -np.where(valid, depth, np.nan)

    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows, cols)
    x = np.where(valid, factor * z * (c - (cols / 2)) / ratio, 0)
    y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, 0)

    return np.dstack((-x, -y, z))


if __name__ == '__main__':
    main()
