import sys

import numpy as np
from pyvtk import PolyData, VtkData, PointData, Scalars, Normals, Vectors
from pathlib import Path


def plot_points(vtk_path, points, normals=None, scalar_fields={}, vector_fields={}):
    structure = PolyData(points=points, vertices=np.arange(len(points)))
    data_fields = []
    if normals is not None:
        data_fields.append(Normals(normals, name='normals'))
    for k, v in scalar_fields.items():
        data_fields.append(Scalars(v, name=k))
    for k, v in vector_fields.items():
        data_fields.append(Vectors(v, name=k))
    if len(data_fields) == 0:
        vtk = VtkData(structure)
    else:
        pointdata = PointData(*data_fields)
        vtk = VtkData(structure, pointdata)
    vtk.tofile(str(vtk_path))
