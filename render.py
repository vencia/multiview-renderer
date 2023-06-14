import bpy
import os
import numpy as np
from pathlib import Path
import math
import argparse
import sys

parser = argparse.ArgumentParser(description='Renders given folder of obj meshes.')
parser.add_argument('--mesh_dir', type=str, default='data/datasets/dmunet/STL_dataset_preprocessed')
parser.add_argument('--render_dir', type=str, default='data/datasets/dmunet/STL_dataset_test_imgs')
parser.add_argument('--num_views', type=int, default=20, choices=[12, 20],
                    help='number of views to be rendered')
parser.add_argument('--fit_view', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--resolution', type=int, default=224,
                    help='Resolution of the images.')
parser.add_argument('--engine', type=str, default='BLENDER_EEVEE',
                    help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def main():
    init_all()

    mesh_dir = Path(args.mesh_dir)
    render_dir = Path(args.render_dir)

    for sample_path in sorted(mesh_dir.rglob('**/*.obj')):
        sample_id = sample_path.stem
        output_folder = render_dir / sample_path.parent.relative_to(mesh_dir) / sample_id
        # if os.path.isdir(output_folder) and len(
        #         [x for x in os.listdir(output_folder) if x.endswith('.png')]) == args.num_views:
        #     continue

        print(sample_id, '...')
        os.makedirs(output_folder, exist_ok=True)

        clear_mesh()

        # # Import textured mesh
        # bpy.ops.object.select_all(action='DESELECT')

        bpy.ops.import_scene.obj(filepath=str(sample_path))

        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

        # Possibly disable specular shading
        for slot in obj.material_slots:
            node = slot.material.node_tree.nodes['Principled BSDF']
            node.inputs['Specular'].default_value = 0.05

        # if args.scale != 1:
        #     bpy.ops.transform.resize(value=(args.scale, args.scale, args.scale))
        #     bpy.ops.object.transform_apply(scale=True)

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
            bpy.context.scene.frame_set(view_id + 1)
            bpy.context.scene.render.filepath = str(output_folder / f'{sample_id}_{view_id:03d}.png')
            render(cam_loc, cam_rot)


def render(cam_loc, cam_rot):
    cam_obj = bpy.context.scene.objects['Camera']
    cam_obj.location = cam_loc
    cam_obj.rotation_euler = cam_rot

    # start rendering
    if args.fit_view:
        bpy.ops.view3d.camera_to_view_selected()
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
    bpy.context.scene.render.engine = args.engine
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
    bpy.context.scene.render.image_settings.color_depth = args.color_depth  # ('8', '16')
    bpy.context.scene.render.image_settings.file_format = args.format  # ('PNG', 'OPEN_EXR', 'JPEG, ...)
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.film_transparent = True

    bpy.context.scene.use_nodes = True
    # bpy.context.scene.view_layers["View Layer"].use_pass_normal = True
    # bpy.context.scene.view_layers["View Layer"].use_pass_diffuse_color = True
    # bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    # Delete default cube
    bpy.context.active_object.select_set(True)
    bpy.ops.object.delete()

    # bpy.ops.object.light_add(type='POINT')
    # lamp_data = bpy.data.lights['POINT']
    # lamp_data.energy = 1.5
    # lamp_object = bpy.data.objects.new(name='l1', object_data=lamp_data)
    # # bpy.context.scene.objects.link(lamp_object)
    # lamp_object.location = (0, 0, 10)
    # lamp_data = bpy.ops.object.light_add(name='l2', type='POINT')
    # lamp_data.energy = 1.5
    # lamp_object = bpy.data.objects.new(name='l2', object_data=lamp_data)
    # # bpy.context.scene.objects.link(lamp_object)
    # lamp_object.location = (0, 0, -10)
    # lamp_data = bpy.ops.object.light_add(name='l3', type='POINT')
    # lamp_data.energy = 0.2
    # lamp_object = bpy.data.objects.new(name='l3', object_data=lamp_data)
    # bpy.context.scene.objects.link(lamp_object)
    # lamp_object.location = (0, 10, 0)
    # lamp_data = bpy.ops.object.light_add(name='l4', type='POINT')
    # lamp_data.energy = 0.2
    # lamp_object = bpy.data.objects.new(name='l4', object_data=lamp_data)
    # bpy.context.scene.objects.link(lamp_object)
    # lamp_object.location = (0, -10, 0)
    # lamp_data = bpy.ops.object.light_add(name='l5', type='POINT')
    # lamp_data.energy = 0.2
    # lamp_object = bpy.data.objects.new(name='l5', object_data=lamp_data)
    # bpy.context.scene.objects.link(lamp_object)
    # lamp_object.location = (10, 0, 0)
    # lamp_data = bpy.ops.object.light_add(name='l6', type='POINT')
    # lamp_data.energy = 0.2
    # lamp_object = bpy.data.objects.new(name='l6', object_data=lamp_data)
    # bpy.context.scene.objects.link(lamp_object)
    # lamp_object.location = (-10, 0, 0)

    # Make light just directional, disable shadows.
    light = bpy.data.lights['Light']
    light.type = 'SUN'
    light.use_shadow = False
    # Possibly disable specular shading:
    light.specular_factor = 1.0
    light.energy = 2.0

    # Add another light source so stuff facing away from light is not completely dark
    bpy.ops.object.light_add(type='SUN')
    light2 = bpy.data.lights['Sun']
    light2.use_shadow = False
    light2.specular_factor = 1.0
    light2.energy = 2.0  # 0.015
    bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
    bpy.data.objects['Sun'].rotation_euler[0] += 180


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


if __name__ == '__main__':
    main()
