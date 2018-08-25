__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports

# Module imports
import os
import shapefile

# Livestock imports
from livestock import geometry


# ---------------------------------------------------------------------------- #
# CMF Functions and Classes


def test_shapely_to_pyshp(shapely_polygons):

    pyshapes = []
    for polygon in shapely_polygons:
        pyshapes.append(geometry.shapely_to_pyshp(polygon))

    assert pyshapes
    for shape in pyshapes:
        assert isinstance(shape, shapefile._Shape)
        assert isinstance(shape.parts, list)
        assert shape.parts[0] == 0
        assert isinstance(shape.points, tuple)
        assert shape.shapeType == 15


def test_obj_to_polygons(obj_file_paths):
    polygon_list = geometry.obj_to_polygons(obj_file_paths)

    assert polygon_list
    assert isinstance(polygon_list, list)

    for polygon in polygon_list:
        assert polygon.is_valid
        assert polygon.type == 'Polygon'
        assert polygon.has_z


def test_centroid_z(shapely_polygons):

    for polygon in shapely_polygons:
        centroid = geometry.centroid_z(polygon)
        assert centroid
        assert isinstance(centroid, float)


def test_obj_to_shp(obj_file_paths, tmpdir):
    test_folder = tmpdir.mkdir('test')
    file = os.path.join(test_folder, 'mesh_shape.shp')
    geometry.obj_to_shp(obj_file_paths, file)

    assert os.path.isfile(file)
    assert os.stat(file).st_size
