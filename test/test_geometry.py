__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports


# Module imports


# Livestock imports
import geometry


# -------------------------------------------------------------------------------------------------------------------- #
# CMF Functions and Classes


def test_shapely_to_pyshp():
    pass


def test_obj_to_polygons(obj_file_paths):
    polygon_list = geometry.obj_to_polygons(obj_file_paths)

    assert polygon_list
    assert isinstance(polygon_list, list)

    for polygon in polygon_list:
        assert polygon.is_valid
        assert polygon.type == 'Polygon'
        assert polygon.has_z


def test_obj_to_shp():
    pass
