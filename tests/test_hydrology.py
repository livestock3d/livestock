__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports

# Module imports
import cmf

# Livestock imports
from livestock import hydrology

# ---------------------------------------------------------------------------- #
# Livestock Test


def test_load_cmf_files(input_files):

    (ground, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = hydrology.load_cmf_files(input_files)

    assert ground
    assert isinstance(ground, list)
    for ground_ in ground:
        assert isinstance(ground_, dict)

    assert mesh_path
    assert isinstance(str(mesh_path), str)

    assert outputs
    assert isinstance(outputs, dict)

    assert solver_settings
    assert isinstance(solver_settings, dict)


def test_mesh_to_cells(obj_file_paths):
    project = cmf.project()

    hydrology.mesh_to_cells(project, obj_file_paths, False)

    assert project
    assert project.cells


def test_create_retention_curve(cmf_data):
    (ground_list, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data

    for ground in ground_list:
        curve_dict = ground['ground_type']['retention_curve']
        r_curve = hydrology.create_retention_curve(curve_dict)

        assert r_curve
        assert isinstance(r_curve, cmf.VanGenuchtenMualem)
        assert r_curve.Ksat == curve_dict['k_sat']
        assert r_curve.alpha == curve_dict['alpha']
        assert r_curve.Phi == curve_dict['phi']
        assert r_curve.n == curve_dict['n']
        assert r_curve.m == curve_dict['m']
        assert r_curve.l == curve_dict['l']


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



