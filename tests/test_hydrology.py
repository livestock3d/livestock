__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import cmf

# Livestock imports
from livestock import hydrology

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Test


def test_load_cmf_files():
    assert True


def test_mesh_to_cells(obj_file_paths):
    project = cmf.project()

    hydrology.mesh_to_cells(project, obj_file_paths, False)

    assert project
    assert project.cells


def test_configure_cells():
    assert True


def test_add_tree_to_project():
    assert True


def test_create_weather():
    assert True


def test_create_boundary_conditions():
    assert True


def test_solve_project():
    assert True


def test_save_project():
    assert True


def test_run_model():
    assert True



