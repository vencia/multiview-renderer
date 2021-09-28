# Multiview Renderer

Simple blender script for rendering shaded images such as used by [MVCNN](https://arxiv.org/abs/1505.00880).

<img src="examples/modelnet_aligned_imgs/airplane_0627/001.png" alt="001.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/002.png" alt="002.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/003.png" alt="003.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/004.png" alt="004.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/005.png" alt="005.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/006.png" alt="006.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/007.png" alt="007.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/008.png" alt="008.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/009.png" alt="009.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/010.png" alt="010.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/011.png" alt="011.png" width="15%" /><img src="examples/modelnet_aligned_imgs/airplane_0627/012.png" alt="012.png" width="15%" />

## Getting Started
- install requirements, preferably in a virtual env
```python
pip install -r requirements.txt
```
- set path to [blender2.79b](https://download.blender.org/release/Blender2.79/) executable in ``settings.py``
- set mesh dataset path in ``settings.py``
## Pre-process Meshes
- convert to .obj format
- normalize
- rotate/flip
```python
python preprocess_meshes.py
```

## Render Views
- render views from a sphere around the object
```python
python render_meshes.py
```

## Links
- MVCNN   https://github.com/jongchyisu/mvcnn_pytorch
- Blender Code https://github.com/weiaicunzai/blender_shapenet_render
