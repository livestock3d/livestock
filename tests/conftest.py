__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports

# Module imports
import pytest
import os
import shutil

# Livestock imports
from livestock import geometry
from livestock import hydrology

# ---------------------------------------------------------------------------- #
# CMF Functions and Classes


@pytest.fixture()
def data_folder():
    parent = os.path.dirname(__file__)
    return os.path.join(parent, 'test_data')


@pytest.fixture(params=[0, 1])
def obj_file_paths(data_folder, request):
    obj_folder = os.path.join(data_folder, 'obj_to_shp')
    return os.path.join(obj_folder, f'mesh_{request.param}.obj')


@pytest.fixture()
def shapely_polygons(obj_file_paths):
    return geometry.obj_to_polygons(obj_file_paths)


@pytest.fixture(params=['run_off'])
def input_files(tmpdir, data_folder, request):

    test_folder = tmpdir.mkdir('test')
    data_path = os.path.join(data_folder, request.param)
    files = os.listdir(data_path)

    for file in files:
        shutil.copyfile(os.path.join(data_path, file),
                        os.path.join(test_folder, file))

    return test_folder


@pytest.fixture()
def cmf_data(input_files):
    return hydrology.load_cmf_files(input_files)
