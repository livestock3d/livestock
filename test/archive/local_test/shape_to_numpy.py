import shapely
import numpy as np
import livestock.geometry as geo

path = r'C:\Users\Christian\Desktop\test_mesh.obj'

shape = geo.obj_to_polygons(path)
point = list(shape[0].exterior.coords)


def centroid_z(polygon):
    z_values = []
    for pt in polygon.exterior.coords:
        z_values.append(pt[2])

    mean_z = sum(z_values)/len(z_values)

    return mean_z
