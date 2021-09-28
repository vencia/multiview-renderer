"""settings.py contains all configuration parameters the blender needs


reference: https://github.com/weiaicunzai/blender_shapenet_render
"""

g_num_views = 12

g_mesh_dataset_path = 'data/datasets/modelnet/modelnet40_manually_aligned'
g_preprocessed_dataset_path = 'data/datasets/modelnet/modelnet40_manually_aligned_preprocessed'
g_imgs_dataset_path = 'data/datasets/modelnet/modelnet40_manually_aligned_imgs'

g_blender_excutable_path = '/home/vencia/blender-2.79b-linux-glibc219-x86_64/blender'

g_fit_camera_to_view = True

# background image composite
# enum in [‘RELATIVE’, ‘ABSOLUTE’, ‘SCENE_SIZE’, ‘RENDER_SIZE’], default ‘RELATIVE’
# g_scale_space = 'RENDER_SIZE'
g_use_film_transparent = True

# camera:
# enum in [‘QUATERNION’, ‘XYZ’, ‘XZY’, ‘YXZ’, ‘YZX’, ‘ZXY’, ‘ZYX’, ‘AXIS_ANGLE’]
g_rotation_mode = 'XYZ'

# enum in [‘BW’, ‘RGB’, ‘RGBA’], default ‘BW’
g_rgb_color_mode = 'BW'
# enum in [‘8’, ‘10’, ‘12’, ‘16’, ‘32’], default ‘8’
g_rgb_color_depth = '8'
g_rgb_file_format = 'PNG'

# engine type [CYCLES, BLENDER_RENDER]
g_engine_type = 'BLENDER_RENDER'

# output image size =  (g_resolution_x * resolution_percentage%, g_resolution_y * resolution_percentage%)
g_resolution_x = 224  # 512
g_resolution_y = 224  # 512
g_resolution_percentage = 100
