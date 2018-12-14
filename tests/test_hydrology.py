__author__ = "Christian Kongsgaard"
__license__ = "GNU GPLv3"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import cmf
import pytest
import os

# Livestock imports
from livestock import hydrology


# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Test


def test_load_cmf_files(input_files):
    (ground, mesh_paths, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = hydrology.load_cmf_files(input_files)

    assert ground
    assert isinstance(ground, list)
    for ground_ in ground:
        assert isinstance(ground_, dict)

    assert mesh_paths
    assert isinstance(mesh_paths, list)
    for mesh in mesh_paths:
        assert isinstance(mesh, str)
        assert os.path.split(mesh)[1].startswith('mesh')
        assert os.path.split(mesh)[1].endswith('.obj')

    assert outputs
    assert isinstance(outputs, dict)

    assert solver_settings
    assert isinstance(solver_settings, dict)


def test_mesh_to_cells(obj_file_paths):
    project = cmf.project()

    hydrology.mesh_to_cells(project, [obj_file_paths, ], False)

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
    (ground_list, mesh_paths, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data

    project, mesh_info = project_with_cells
    for ground in ground_list:
        for cell_index in mesh_info[ground['mesh']]:
            hydrology.build_cell(cell_index, project,
                                 ground, retention_curve)

            cell = project.cells[cell_index]

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
    project, mesh_info = project_with_cells

    for ground in ground_list:
        hydrology.install_flux_connections(project, ground)

    # TODO: Create better assessments
    assert project


def test_configure_cells(cmf_data, project_with_cells):
    (ground_list, mesh_path, weather_dict, trees_dict, outputs,
     solver_settings, boundary_dict) = cmf_data
    project, mesh_info = project_with_cells

    for ground in ground_list:
        hydrology.configure_cells(project, ground,  mesh_info[ground['mesh']])

    # TODO: Create better assessments
    assert project_with_cells


def test_add_tree_to_project():
    assert True


def test_create_time_series(get_weather_data):

    temperature = get_weather_data['weather']['temp']['all']
    time_series = hydrology.create_time_series(temperature, get_weather_data['settings'])

    assert isinstance(time_series, cmf.timeseries)
    assert len(temperature) == len(time_series)

    for i in range(len(temperature)):
        assert temperature[i] == time_series[i]


def test_weather_to_time_series(get_weather_data):
    weather = get_weather_data['weather']
    settings = get_weather_data['settings']
    test_weather = {}

    for weather_type in weather.keys():
        if weather_type in ['temp', 'sun']:
            test_weather[weather_type] = None
        else:
            # Try for weather type having the same weather for all cells
            try:
                test_weather[weather_type] = weather[weather_type]['all']

            # Accept latitude, longitude and time zone
            except TypeError:
                pass

    weather_series = hydrology.weather_to_time_series(test_weather, settings)

    assert weather_series
    assert isinstance(weather_series, dict)
    assert not weather_series['temp']
    assert isinstance(weather_series['wind'], cmf.timeseries)
    assert isinstance(weather_series['rel_hum'], cmf.timeseries)
    assert not weather_series['sun']
    assert isinstance(weather_series['rad'], cmf.timeseries)
    assert isinstance(weather_series['rain'], cmf.timeseries)
    assert isinstance(weather_series['ground_temp'], cmf.timeseries)


def test_get_weather_for_cell(get_weather_data):

    cell_weather = hydrology.get_weather_for_cell(0, get_weather_data['weather'], get_weather_data['settings'])

    assert cell_weather
    assert isinstance(cell_weather, tuple)
    assert isinstance(cell_weather[0], dict)
    assert isinstance(cell_weather[1], dict)

    location_keys = ['time_zone', 'latitude', 'longitude']
    for key in cell_weather[1].keys():
        assert key in location_keys


def test_create_weather(get_project_and_weather_data):

    project, weather, solver_settings = get_project_and_weather_data
    hydrology.create_weather(project, weather, solver_settings)

    assert project.meteo_stations


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


def test_solve_project(solve_ready_project, mock_solver, mock_gather_results):
    results = hydrology.solve_project(*solve_ready_project)

    assert results


def test_save_project():
    assert True


@pytest.mark.skip('Not ready yet')
def test_run_off(data_folder):
    folder = os.path.join(data_folder, 'run_off')
    hydrology.run_model(folder)
