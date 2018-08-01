__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports

# Module imports
import cmf
import pytest

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


@pytest.mark.skip('Something wrong with the setup')
def test_install_cell_connections(cmf_data, project_with_cells):
    (ground_list, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data

    for ground in ground_list:
        for cell_index in ground['mesh']:
            cell = project_with_cells.cells[cell_index]
            hydrology.install_cell_connections(cell, ground['et_method'])


def test_build_cell(cmf_data, project_with_cells, retention_curve):
    (ground_list, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data

    for ground in ground_list:
        for cell_index in ground['mesh']:
            hydrology.build_cell(cell_index, project_with_cells,
                                 ground, retention_curve)

            cell = project_with_cells.cells[cell_index]

            assert cell
            if len(cell.layers) > 1:
                assert len(cell.layers) == len(ground['mesh']['layers'])
            assert cell.evaporation
            assert cell.transpiration
            assert cell.vegetation
            assert pytest.approx(cell.saturated_depth == ground['ground_type'][
                'saturated_depth'])

    assert True


def test_install_flux_connections(cmf_data, project_with_cells):
    (ground_list, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data

    for ground in ground_list:
        hydrology.install_flux_connections(project_with_cells, ground)

    # TODO: Create better assessments
    assert project_with_cells


def test_configure_cells(cmf_data, project_with_cells):
    (ground_list, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data

    for ground in ground_list:
        hydrology.configure_cells(project_with_cells, ground)

    # TODO: Create better assessments
    assert project_with_cells


def test_add_tree_to_project():
    assert True


def test_create_weather():
    assert True


def test_create_boundary_conditions():
    assert True


def test_config_outputs():
    assert True


def test_gather_results():
    assert True


def test_get_analysis_length():
    assert True


def test_get_time_step():
    assert True


@pytest.mark.skip('Not yet finished')
def test_solve_project(solve_ready_project, mock_solver, mock_gather_results):
    results = hydrology.solve_project(*solve_ready_project)

    assert results


def test_save_project():
    assert True


def test_run_model():
    assert True
