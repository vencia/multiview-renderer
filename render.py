import bpy
import os
import numpy as np
import glob
from pathlib import Path
import math
import random
from settings import *


def main():
    init_all()

    for sample_path in glob.glob(g_preprocessed_dataset_path + '/**/*.obj', recursive=True):
        sample_id = Path(sample_path).stem
        print(sample_id)
        output_dir = str(
            g_imgs_dataset_path / Path(sample_path).parent.relative_to(g_preprocessed_dataset_path) / sample_id)
        os.makedirs(output_dir, exist_ok=True)

        clear_mesh()
        bpy.ops.import_scene.obj(filepath=sample_path)
        img_file_output_node = bpy.context.scene.node_tree.nodes[4]
        img_file_output_node.base_path = output_dir

        azimuths = np.linspace(0, 2 * np.pi, g_n_views, endpoint=False)
        elevation = np.pi / 4
        distance = math.sqrt(3) * 0.8
        tilt = 0.0

        for view_id in range(g_n_views):
            azimuth = azimuths[view_id]
            cam_loc = camera_location(azimuth, elevation, distance)
            cam_rot = camera_rot_XYZEuler(azimuth, elevation, tilt)
            bpy.context.scene.frame_set(view_id + 1)
            # bpy.context.scene.render.filepath = output_dir + '/' + sample_id + '_' + str(view_id) + '.png'
            render(cam_loc, cam_rot)


def render(cam_loc, cam_rot):
    cam_obj = bpy.data.objects['Camera']
    cam_obj.location = cam_loc
    cam_obj.rotation_euler = cam_rot

    bpy.context.scene.render.alpha_mode = 'TRANSPARENT'

    img_file_output_node = bpy.context.scene.node_tree.nodes[4]
    img_file_output_node.file_slots[0].path = '###.png'

    # start rendering
    if g_fit_camera_to_view:
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
            obj.select = True
    bpy.ops.object.delete()


def scene_setting_init():
    """initialize blender setting configurations

    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = g_engine_type
    bpy.data.scenes[sce].cycles.film_transparent = g_use_film_transparent

    # dimensions
    bpy.data.scenes[sce].render.resolution_x = g_resolution_x
    bpy.data.scenes[sce].render.resolution_y = g_resolution_y
    bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage


def node_setting_init():
    """node settings for render rgb images

    mainly for compositing the background images
    """

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    img_file_output_node = tree.nodes.new('CompositorNodeOutputFile')

    img_file_output_node.format.color_mode = g_rgb_color_mode
    img_file_output_node.format.color_depth = g_rgb_color_depth
    img_file_output_node.format.file_format = g_rgb_file_format

    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
    links.new(alpha_over_node.outputs[0], img_file_output_node.inputs[0])


def camera_setting_init():
    """ camera settings for renderer
    """
    bpy.data.objects['Camera'].rotation_mode = g_rotation_mode


def light_setting_init():
    lamp_data = bpy.data.lamps.new(name="New Lamp", type='HEMI')
    lamp_data.energy = 0.1
    lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
    bpy.context.scene.objects.link(lamp_object)
    lamp_object.location = (0, 0, 0)
    lamp_object.select = True
    bpy.context.scene.objects.active = lamp_object


def init_all():
    """init everything we need for rendering
    an image
    """
    scene_setting_init()
    camera_setting_init()
    node_setting_init()
    light_setting_init()


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


if __name__ == '__main__':
    main()
