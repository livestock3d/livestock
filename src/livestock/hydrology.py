__author__ = "Christian Kongsgaard"
__license__ = "GNU GPLv3"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import cmf
from cmf.geos_shapereader import Shapefile
import cmf.geometry
import datetime
import numpy as np
import os
import ast
import json
import typing
import progressbar

# Livestock imports
from livestock import geometry as lg
from livestock.log import livestock_logger

logger = livestock_logger()


# -------------------------------------------------------------------------------------------------------------------- #
# CMF Functions and Classes


def mesh_to_cells(cmf_project: cmf.project, mesh_paths: typing.List[str],
                  delete_after_load=False) -> typing.Tuple[cmf.project, dict]:
    """
    Takes a mesh and converts it into CMF cells

    :param mesh_paths: Path to mesh .obj file
    :param cmf_project: CMF project object.
    :param delete_after_load: If True, it deletes the input files after they have been loaded.
    :return: True
    """

    cell_dict = {}
    cell_counter = 0
    for mesh_file in mesh_paths:
        mesh_name = os.path.split(mesh_file)[1][:-4]
        cell_dict[mesh_name] = list()
        # Convert obj to shapefile
        shape_path = os.path.split(mesh_file)[0] + '/mesh.shp'
        lg.obj_to_shp(mesh_file, shape_path)
        polygons = Shapefile(shape_path)
        logger.debug('Converted .obj to .shp')

        for polygon in polygons:
            cmf.geometry.create_cell(cmf_project, polygon.shape,
                                     polygon.height, polygon.id)
            cell_dict[mesh_name].append(cell_counter)
            cell_counter += 1

        # Build topology
        cmf.geometry.mesh_project(cmf_project, verbose=False)
        logger.info('Build cells in project')

        if delete_after_load:
            os.remove(mesh_file)
            os.remove(shape_path)
            os.remove(os.path.split(mesh_file)[0] + '/mesh.dbf')
            os.remove(os.path.split(mesh_file)[0] + '/mesh.shx')
            logger.debug('Removed mesh files')

    return cmf_project, cell_dict


def set_vegetation_properties(cell: cmf.Cell, property_dict: dict) -> cmf.Cell:
    """
    Sets the vegetation properties for a cell.

    :param cell: Cell to set properties for.
    :param property_dict: Dict containing the needed properties.
    :return: modified project
    """

    cell.vegetation.Height = float(property_dict['height'])
    cell.vegetation.LAI = float(property_dict['lai'])
    cell.vegetation.albedo = float(property_dict['albedo'])
    cell.vegetation.CanopyClosure = float(property_dict['canopy_closure'])
    cell.vegetation.CanopyParExtinction = float(property_dict['canopy_par'])
    cell.vegetation.CanopyCapacityPerLAI = float(property_dict['canopy_capacity'])
    cell.vegetation.StomatalResistance = float(property_dict['stomatal_res'])
    cell.vegetation.RootDepth = float(property_dict['root_depth'])
    cell.vegetation.fraction_at_rootdepth = float(property_dict['root_fraction'])
    cell.vegetation.LeafWidth = float(property_dict['leaf_width'])

    logger.debug(f'Sat vegetation properties for cell at: {cell.get_position()}')

    return cell


def add_tree_to_project(project: cmf.project, cell_index: int, properties: dict) -> cmf.project:
    """
    Adds a tree to the model and sets the need properties for it.

    :param project: CMF project
    :param cell_index: Index of the cell where the tree should be added.
    :param properties: Dict with tree properties.
    :return: modified project
    """

    cell = project.cells[cell_index]
    set_vegetation_properties(cell, properties)
    name = 'canopy_' + str(cell_index)
    cell.add_storage(name, 'C')

    cmf.Rainfall(cell.canopy, cell, False, True)
    cmf.Rainfall(cell.surfacewater, cell, True, False)
    cmf.RutterInterception(cell.canopy, cell.surfacewater, cell)
    cmf.CanopyStorageEvaporation(cell.canopy, cell.evaporation, cell)

    logger.debug(f'Added a tree to cell at: {cell.get_position()}')

    return project


def load_cmf_files(folder: str, delete_after_load=False) -> tuple:
    """
    Loads the needed files for the CMF project to run.

    :param folder: Folder where the input files are located
    :param delete_after_load: Delete after the files are loaded?
    :return: True
    """

    # Load files and assign data to variables
    ground = load_ground(folder, delete_after_load)
    mesh_paths = load_meshes(folder)

    weather_dict = load_input_data(os.path.join(folder, 'weather.json'),
                                   delete_after_load)
    trees_dict = load_input_data(os.path.join(folder, 'tree.json'),
                                 delete_after_load)
    outputs = load_input_data(os.path.join(folder, 'outputs.json'),
                              delete_after_load)
    solver_settings = load_input_data(os.path.join(folder, 'solver.json'),
                                      delete_after_load)
    boundary_dict = load_input_data(os.path.join(folder,
                                                 'boundary_condition.json'),
                                    delete_after_load)

    logger.info('Loaded input files')

    return (ground, mesh_paths, weather_dict, trees_dict, outputs,
            solver_settings, boundary_dict)


def load_input_data(path: str, delete: bool) -> typing.Optional[dict]:
    # look for file
    if os.path.isfile(path):

        with open(path, 'r') as file:
            input_dict = json.load(file)

        # delete file
        if delete:
            os.remove(path)

        return input_dict

    else:
        logger.info(f'{os.path.split(path)[1]} not found')
        return None


def load_ground(folder: str, delete: bool) -> list:
    # look for file
    if os.path.isfile(folder + '/ground.json'):
        ground_path = folder + '/ground.json'

        with open(ground_path, 'r') as file:
            ground = json.load(file)

        # delete file
        if delete:
            os.remove(ground_path)

        return ground

    else:
        logger.error(f'Cannot find ground.json in folder: {folder}')
        raise FileNotFoundError(f'Cannot find ground.json in folder: '
                                f'{folder}')


def load_meshes(folder: str) -> typing.List[str]:
    # look for file
    meshes = []
    for file in os.listdir(folder):
        if file.endswith('.obj'):
            meshes.append(os.path.join(folder, file))

    if meshes:
        return meshes
    else:
        error = f'Cannot find any .obj files in folder: {folder}'
        logger.error(error)
        raise FileNotFoundError(error)


def create_retention_curve(retention_curve: dict) -> cmf.VanGenuchtenMualem:
    """
    Converts a dict of retention curve parameters into a CMF van
    Genuchten-Mualem retention curve.

    :param retention_curve: dict
    :return: CMF retention curve
    """

    curve = cmf.VanGenuchtenMualem(retention_curve['k_sat'],
                                   retention_curve['phi'],
                                   retention_curve['alpha'],
                                   retention_curve['n'],
                                   retention_curve['m'])
    curve.l = retention_curve['l']

    logger.debug(f'Created retention curve.')

    return curve


def install_cell_connections(cell: cmf.Cell, evapotranspiration_method: str) \
        -> cmf.Cell:
    # Install connections
    cell.install_connection(cmf.Richards)
    cell.install_connection(cmf.SimpleInfiltration)

    if evapotranspiration_method == 'penman_monteith':
        # Install Penman & Monteith method to calculate evapotranspiration
        # potential
        cell.install_connection(cmf.PenmanMonteithET)

        # Install surface water evaporation
        cmf.PenmanEvaporation(cell.surfacewater, cell.evaporation,
                              cell.meteorology)

        logger.debug(f'Install Richards connection, simple infiltration, '
                     f'Penman-Monteith evapotranspiration and '
                     f'Penman evaporation for cell at: {cell.get_position()}')

    elif evapotranspiration_method == 'shuttleworth_wallace':
        # Install Shuttleworth-Wallace method to calculate evapotranspiration
        cell.install_connection(cmf.ShuttleworthWallace)

        # Install surface water evaporation
        cmf.PenmanEvaporation(cell.surfacewater, cell.evaporation,
                              cell.meteorology)

        logger.debug(f'Install Richards connection, simple infiltration, '
                     f'Shuttleworth-Wallace evapotranspiration and '
                     f'Penman evaporation for cell at: {cell.get_position()}')

    return cell


def install_flux_connections(cmf_project: cmf.project, cell_properties: dict) -> cmf.project:
    cmf.connect_cells_with_flux(cmf_project, cmf.DarcyKinematic)
    logger.info('Installed Darcy kinematic for all cells in project.')

    if cell_properties['runoff_method'] == 'kinematic':
        cmf.connect_cells_with_flux(cmf_project, cmf.KinematicSurfaceRunoff)
        logger.info('Installed kinematic surface run-off')

    elif cell_properties['runoff_method'] == 'diffusive':
        cmf.DiffusiveSurfaceRunoff.set_linear_slope(1e-8)
        cmf.connect_cells_with_flux(cmf_project, cmf.DiffusiveSurfaceRunoff)
        logger.info('Installed diffusive surface run-off. Slope set to 1e-8.')

    return cmf_project


def build_cell(cell_id: int, cmf_project: cmf.project, cell_properties: dict,
               r_curve: cmf.VanGenuchtenMualem) -> cmf.project:
    cell = cmf_project.cells[cell_id]
    logger.debug(f'Building cell with ID: {cell_id}')

    # Add layers
    for i in range(len(cell_properties['layers'])):
        cell.add_layer(float(cell_properties['layers'][i]), r_curve)

    install_cell_connections(cell, cell_properties['et_method'])

    set_vegetation_properties(cell, cell_properties['ground_type']['surface_properties'])

    if cell_properties['ground_type']['manning']:
        cell.surfacewater.set_nManning(cell_properties['ground_type']['manning'])
        logger.debug(f'Sat Manning roughness to: {cell_properties["ground_type"]["manning"]}')

    if cell_properties['ground_type']['puddle_depth']:
        cell.surfacewater.puddledepth = cell_properties['ground_type']['puddle_depth']
        logger.debug(f'Sat puddle depth to: {cell_properties["ground_type"]["puddle_depth"]}')

    if cell_properties['surface_water']:
        cell.surfacewater.volume = cell_properties['surface_water']
        logger.info(f'Sat surface water to {cell_properties["surface_water"]} for cell with ID: {cell_id}')

    # Set initial saturation
    cell.saturated_depth = cell_properties['ground_type']['saturated_depth']
    logger.debug(f'Sat saturated depth to {cell_properties["ground_type"]["saturated_depth"]}')

    return cmf_project


def configure_cells(cmf_project: cmf.project, cell_properties_dict: dict, mesh_info: dict) -> cmf.project:
    """
    Configure and setup all needed information for the cells.

    :param cmf_project: CMF project
    :param cell_properties_dict: Dict with all the needed properties
    :return: cmf project
    """

    # Helper functions

    # Convert retention curve parameters into CMF retention curve
    r_curve = create_retention_curve(cell_properties_dict['ground_type']['retention_curve'])

    for cell_index in mesh_info:
        build_cell(cell_index, cmf_project, cell_properties_dict, r_curve)

    # Connect fluxes
    install_flux_connections(cmf_project, cell_properties_dict)

    logger.info('Configured all cells in project.')

    return cmf_project


def config_outputs(cmf_project: cmf.project, output_configuration: dict) -> dict:
    """
    Function to set up result gathering dictionary.

    :param cmf_project: CMF project to collect from
    :param output_configuration: Dict to configure the results from
    :return: Empty result dictionary
    """

    results = {}

    for cell_index in range(0, len(cmf_project.cells)):
        cell_name = 'cell_' + str(cell_index)
        results[cell_name] = {}

        # Set all cell related outputs
        for cell_output in output_configuration['cell']:
            results[cell_name][str(cell_output)] = []

        for layer_index in range(0, len(cmf_project.cells[cell_index].layers)):
            layer_name = 'layer_' + str(layer_index)
            results[cell_name][layer_name] = {}

            # Set all layer related outputs
            for layer_output in output_configuration['layer']:
                results[cell_name][layer_name][layer_output] = []

    logger.debug(f'Configured outputs for CMF project. Outputs was: {results}')

    return results


def gather_results(cmf_project: cmf.project, results: dict, time: datetime.datetime):
    """
    Collects the produced results.

    :param cmf_project: CMF project to collect from
    :param results: Dict to gather the results in
    :param time: Point in solver time
    :return:
    """

    logger.debug(f'Gathering results for time: {time}')

    for cell_index in range(0, len(cmf_project.cells)):
        cell_name = 'cell_' + str(cell_index)

        for out_key in results[cell_name].keys():

            # Collect cell related results
            if not out_key.startswith('layer'):
                if out_key == 'evaporation':
                    evap = cmf_project.cells[cell_index].evaporation

                    flux_at_time = 0
                    for flux, node in evap.fluxes(time):
                        flux_at_time += flux

                    results[cell_name][out_key].append(flux_at_time)

                elif out_key == 'transpiration':
                    transp = cmf_project.cells[cell_index].transpiration

                    flux_at_time = 0
                    for flux, node in transp.fluxes(time):
                        flux_at_time += flux

                    results[cell_name][out_key].append(flux_at_time)

                elif out_key == 'surface_water_volume':
                    volume = cmf_project.cells[cell_index].surfacewater.volume
                    results[cell_name][out_key].append(volume)

                elif out_key == 'surface_water_flux':
                    water = cmf_project.cells[cell_index].get_surfacewater()

                    flux_and_node = []
                    for flux, node in water.fluxes(time):
                        flux_and_node.append((flux, node))

                    results[cell_name][out_key].append(flux_and_node)

                elif out_key == 'heat_flux':
                    results[cell_name][out_key].append(cmf_project.cells[cell_index].heat_flux(time))

                elif out_key == 'aerodynamic_resistance':
                    results[cell_name][out_key].append(
                        cmf_project.cells[cell_index].get_aerodynamic_resistance(time))

        for layer_index in range(0, len(cmf_project.cells[cell_index].layers)):
            layer_name = 'layer_' + str(layer_index)

            for out_key in results[cell_name][layer_name].keys():

                # Collect layer related results

                if out_key == 'potential':
                    results[cell_name][layer_name][out_key].append(
                        cmf_project.cells[cell_index].layers[layer_index].potential)

                elif out_key == 'theta':
                    results[cell_name][layer_name][out_key].append(
                        cmf_project.cells[cell_index].layers[layer_index].theta)

                elif out_key == 'volumetric_flux':
                    layer = cmf_project.cells[cell_index].layers[layer_index].get_3d_flux(time)

                    """
                    flux_and_node = []
                    for flux, node in layer.fluxes(time):
                        flux_and_node.append((flux, node))
                    """

                    results[cell_name][layer_name][out_key].append(layer)

                elif out_key == 'volume':
                    results[cell_name][layer_name][out_key].append(
                        cmf_project.cells[cell_index].layers[layer_index].volume)

                elif out_key == 'wetness':
                    results[cell_name][layer_name][out_key].append(
                        cmf_project.cells[cell_index].layers[layer_index].wetness)


def get_analysis_length(analysis_length: list) -> datetime.timedelta:
    length, quantity = analysis_length

    if quantity == 'y':
        logger.debug(f'Analysis length set to: {length} {quantity}')
        return datetime.timedelta(days=length * 365)

    elif quantity == 'd':
        logger.debug(f'Analysis length set to: {length} {quantity}')
        return datetime.timedelta(days=length)

    elif quantity == 'h':
        logger.debug(f'Analysis length set to: {length} {quantity}')
        return datetime.timedelta(hours=length)

    elif quantity == 'm':
        logger.debug(f'Analysis length set to: {length} {quantity}')
        return datetime.timedelta(minutes=length)

    elif quantity == 's':
        logger.debug(f'Analysis length set to: {length} {quantity}')
        return datetime.timedelta(seconds=length)


def get_time_step(time_step: list) -> datetime.timedelta:
    step_size, quantity = time_step

    if quantity == 'y':
        logger.debug(f'Solver time step set to: {step_size} {quantity}')
        return datetime.timedelta(days=step_size * 365)

    elif quantity == 'd':
        logger.debug(f'Solver time step set to: {step_size} {quantity}')
        return datetime.timedelta(days=step_size)

    elif quantity == 'h':
        logger.debug(f'Solver time step set to: {step_size} {quantity}')
        return datetime.timedelta(hours=step_size)

    elif quantity == 'm':
        logger.debug(f'Solver time step set to: {step_size} {quantity}')
        return datetime.timedelta(minutes=step_size)

    elif quantity == 's':
        logger.debug(f'Solver time step set to: {step_size} {quantity}')
        return datetime.timedelta(seconds=step_size)


def solve_project(cmf_project: cmf.project, solver_settings: dict,
                  outputs: dict) -> dict:
    """Solves the model"""

    logger.info('Initializing solving of CMF project')

    # Create solver, set time and set up results
    solver = cmf.CVodeIntegrator(cmf_project,
                                 solver_settings['tolerance'])
    solver.t = cmf.Time(solver_settings['start_time']['day'],
                        solver_settings['start_time']['month'],
                        solver_settings['start_time']['year'])

    logger.debug(f'Solver start time: {solver.t}')

    results = config_outputs(cmf_project, outputs)

    # Save initial conditions to results
    gather_results(cmf_project, results, solver.t)

    analysis_length = get_analysis_length(solver_settings['analysis_length'])
    time_step = get_time_step(solver_settings['time_step'])
    number_of_steps = analysis_length.total_seconds() / time_step.total_seconds()

    # Run solver and save results at each time step
    widgets = [' [', progressbar.Timer(), '] ',
               progressbar.Bar(),
               ' [', progressbar.AdaptiveETA(), ' ]']
    bar = progressbar.ProgressBar(max_value=number_of_steps, widgets=widgets)
    for index, time_ in enumerate(solver.run(solver.t,
                                             solver.t + analysis_length,
                                             time_step)):
        gather_results(cmf_project, results, time_)
        bar.update(index)

    logger.info('Solved CMF project!')

    return results


def save_results(results: dict, folder: str):
    """Saves the computed results to a json file"""

    path = os.path.join(folder, 'results.json')

    logger.debug(f'Saving results to: {path}')

    with open(path, 'w') as file:
        json.dump(results, file)


def create_time_series(weather_data: list, solver_settings: dict) -> typing.Optional[cmf.timeseries]:

    if weather_data:
        start = cmf.Time(solver_settings['start_time']['day'],
                         solver_settings['start_time']['month'],
                         solver_settings['start_time']['year'])
        step = get_time_step(solver_settings['time_step'])

        # Create time series
        return cmf.timeseries.from_array(start, step, np.asarray(weather_data))

    else:
        return None


def weather_to_time_series(weather: dict, solver_settings: dict) -> dict:

    time_series = {}
    # Create time series
    for weather_key in weather.keys():
        try:
            series = create_time_series(weather[weather_key], solver_settings)
            time_series[weather_key] = series
        except KeyError:
            time_series[weather_key] = None

    return time_series


def get_weather_for_cell(cell_id: int, project_weather: dict, project_settings: dict) -> typing.Tuple[dict, dict]:
    # Initialize
    cell_weather_dict_ = {}
    location_dict = {}

    # Find weather matching cell ID
    for weather_type in project_weather.keys():
        # Try for weather type having the same weather for all cells
        try:
            cell_weather_dict_[weather_type] = project_weather[weather_type]['all']

        # Accept that some have one for each cell
        except KeyError:
            cell_weather_dict_[weather_type] = project_weather[weather_type]['cell_' + str(cell_id)]

        # Accept latitude, longitude and time zone
        except TypeError:
            if weather_type in ['time_zone', 'latitude', 'longitude']:
                location_dict[weather_type] = project_weather[weather_type]

    # Convert to time series
    cell_weather_series = weather_to_time_series(cell_weather_dict_, project_settings)

    return cell_weather_series, location_dict


def create_weather_station(project: cmf.project, cell_id: int, weather: dict, location: dict) -> tuple:

    # Add cell rainfall station to the project
    try:
        weather['rain']
        rain_station = project.rainfall_stations.add(Name=f'cell_{cell_id} rain',
                                                          Data=weather['rain'],
                                                          Position=(0, 0, 0))
    except KeyError:
        rain_station = None

    # Add cell meteo station to the project
    meteo_station = project.meteo_stations.add_station(name=f'cell_{cell_id} weather',
                                                            position=(0, 0, 0),
                                                            latitude=location['latitude'],
                                                            longitude=location['longitude'],
                                                            tz=location['time_zone'])

    try:
        meteo_station.T = weather['temp']
        meteo_station.Tmax = meteo_station.T.reduce_max(meteo_station.T.begin, cmf.day)
        meteo_station.Tmin = meteo_station.T.reduce_min(meteo_station.T.begin, cmf.day)
    except KeyError:
        pass

    try:
        meteo_station.Windspeed = weather['wind']
    except KeyError:
        pass

    try:
        meteo_station.rHmean = weather['rel_hum']
    except KeyError:
        pass

    try:
        meteo_station.Sunshine = weather['sun']
    except KeyError:
        pass

    try:
        meteo_station.Rs = weather['rad']
    except KeyError:
        pass

    try:
        meteo_station.Tground = weather['ground_temp']
    except KeyError:
        pass

    return rain_station, meteo_station


def connect_weather_to_cell(cell: cmf.Cell, rain_station, meteo_station) -> None:

    meteo_station.use_for_cell(cell)

    if rain_station:
        rain_station.use_for_cell(cell)


def create_weather(project: cmf.project, weather: dict, solver_settings: dict) -> None:

    for cell_index in range(len(project.cells)):
        cell = project.cells[cell_index]

        cell_weather_dict, project_location = get_weather_for_cell(cell_index, weather, solver_settings)
        cell_rain, cell_meteo = create_weather_station(project, cell_index, cell_weather_dict, project_location)
        connect_weather_to_cell(cell, cell_rain, cell_meteo)


def create_boundary_conditions(cmf_project):
    raise NotImplementedError


def run_model(folder: str) -> cmf.project:
    """
    Runs the model with everything.

    :return: Simulated CMF project
    """

    # Initialize project
    project = cmf.project()

    (ground_list, mesh_paths, weather_dict, trees_dict, outputs, solver_settings,
     boundary_dict) = load_cmf_files(folder, False)

    # Add cells and properties to them
    project, mesh_info = mesh_to_cells(project, mesh_paths)

    for ground in ground_list:
        configure_cells(project, ground, mesh_info[ground['mesh']])

    if trees_dict:
        for key in trees_dict.keys():
            add_tree_to_project(project,
                                trees_dict[key]['face_index'],
                                trees_dict[key]['property'])

    # Create the weather
    if weather_dict:
        create_weather(project, weather_dict, solver_settings)

    # Create boundary conditions
    if boundary_dict:
        create_boundary_conditions(project)

    # Run solver
    results = solve_project(project, solver_settings, outputs)

    # Save the results
    save_results(results, folder)

    return project


def cmf_results(path: str) -> bool:
    """
    Process a CMF result file.

    :param path: Folder containing the result file.
    :type path: str
    :return: True
    :rtype: bool
    """

    files = os.listdir(path)
    result_path = None
    lookup_path = None

    for f in files:
        if f.startswith('results'):
            result_path = path + '/' + f
        elif f.startswith('result_lookup'):
            lookup_path = path + '/' + f

    # Read look up file
    file_obj = open(lookup_path, 'r')
    line = file_obj.readline()
    lookup_dict = eval(line)

    for lookup_key in lookup_dict.keys():
        if lookup_key.startswith('cell'):
            cell_results(lookup_dict[lookup_key], result_path, path)
        elif lookup_key.startswith('layer'):
            layer_results(lookup_dict[lookup_key], result_path, path)

    return True


def cell_results(looking_for: str, result_file: str, folder: str) -> bool:
    """Processes cell results after a desired parameter.

    :param looking_for: Parameter to look for.
    :type looking_for: str
    :param result_file: Path of the result file.
    :type result_file: str
    :param folder: Path of the folder
    :type folder: str
    :return: True
    :rtype: bool
    """

    # Initialize
    result_tree = ET.tostring(ET.parse(result_file).getroot())
    results = xmltodict.parse(result_tree)
    results_to_save = []

    # Find results
    for cell in results['result'].keys():
        if looking_for == 'evapotranspiration':
            evapo = np.array(eval(results['result'][cell]['evaporation']))
            transp = np.array(eval(results['result'][cell]['transpiration']))
            evapotransp = evapo + transp
            results_to_save.append(list(evapotransp))

        else:
            for result in results['result'][cell]:
                if result == looking_for:
                    if result == 'heat_flux':
                        # Covert heat flux from MJ/(m2*day) to W/m2h
                        flux_mj = np.array(eval(results['result'][cell][result]))
                        flux_wh = flux_mj / 0.0864
                        results_to_save.append(list(flux_wh))

                    else:
                        results_to_save.append(eval(results['result'][cell][result]))

    # Write files
    file_path = folder + '/' + looking_for + '.csv'
    csv_file = open(file_path, 'w')
    for result_ in results_to_save:
        csv_file.write(','.join(str(r)
                                for r in result_) + '\n')
    csv_file.close()

    return True


def layer_results(looking_for: str, result_file: str, folder: str) -> bool:
    """
    Processes layer results after a desired parameter.

    :param looking_for: Parameter to look for.
    :type looking_for: str
    :param result_file: Path of the result file.
    :type result_file: str
    :param folder: Path of the folder
    :type folder: str
    :return: True
    :rtype: bool
    """

    # Initialize
    result_tree = ET.tostring(ET.parse(result_file).getroot())
    results = xmltodict.parse(result_tree)
    results_to_save = []

    # find results
    for cell in results['result'].keys():

        for cell_result in results['result'][cell].keys():
            if cell_result.startswith('layer'):
                for layer_result in results['result'][cell][cell_result]:
                    if layer_result == looking_for:
                        if layer_result == 'volumetric_flux':
                            results_to_save.append(convert_cmf_points(results['result']
                                                                      [cell]
                                                                      [cell_result]
                                                                      [layer_result]))

                        else:
                            results_to_save.append(eval(results['result'][cell][cell_result][layer_result]))

        results_to_save.append([cell])

    # Write files
    file_path = folder + '/' + looking_for + '.csv'
    csv_file = open(file_path, 'w')
    for result_ in results_to_save:
        csv_file.write(','.join(str(r) for r in result_) + '\n')
    csv_file.close()

    return True


def convert_cmf_points(points: str) -> list:
    """
    Convert a string of CMF points into a list of tuples with their coordinates.

    :param points: CMF points
    :type points: str
    :return: List of tuples contain point coordniates
    :rtype: list
    """

    # convert to list
    point_list = points[1:-1]
    point_tuples = []

    for tup in point_list.split(' '):
        try:
            tup1 = ast.literal_eval(tup[9:-1])
        except SyntaxError:
            tup1 = ast.literal_eval(tup[9:])
        point_tuples.append(' '.join(str(e) for e in tup1))

    return point_tuples


def surface_flux_results(path: str) -> bool:
    """
    Generates the surface flux from a CMF result file.

    :param path: Folder where the result file is located.
    :type path: str
    :return: True
    :rtype: bool
    """

    # Helper functions
    def read_files(path_):

        # get flux configuration file
        if os.path.isfile(path_ + '/flux_config.txt'):
            file_obj = open(path_ + '/flux_config.txt', 'r')
            flux_lines = file_obj.readlines()
            file_obj.close()
            os.remove(path_ + '/flux_config.txt')

            flux = {'run_off': ast.literal_eval(flux_lines[0].strip()),
                    'rain': ast.literal_eval(flux_lines[1].strip()),
                    'evapotranspiration': ast.literal_eval(flux_lines[2].strip()),
                    'infiltration': ast.literal_eval(flux_lines[3].strip())
                    }
        else:
            raise FileNotFoundError('Cannot find flux_config.txt in folder: ' + str(path_))

        # get center points file
        if os.path.isfile(path_ + '/center_points.txt'):
            file_obj = open(path_ + '/center_points.txt', 'r')
            point_lines = file_obj.readlines()
            file_obj.close()
            os.remove(path_ + '/center_points.txt')

            # process points
            center_points_ = []
            for line_ in point_lines:
                point_ = []
                for p in line_.strip().split(','):
                    point_.append(float(p))
                center_points_.append(np.array(point_))

        else:
            raise FileNotFoundError('Cannot find center_points.txt in folder: ' + str(path_))

        return flux, center_points_

    def get_flux_result(path_: str) -> dict:

        # helper
        def convert_to_list(string_: str) -> list:

            time_steps = []
            current_time = []
            flux_pair = []
            for element in string_[1:-1].split(','):
                if element.startswith('['):
                    flux_pair.append(float(element[2:]))

                elif element.startswith(' ['):
                    flux_pair.append(float(element[3:]))

                elif element.startswith(' ('):
                    flux_pair.append(float(element[2:]))

                elif element.endswith(')'):
                    flux_pair.append(element[1:-1])
                    current_time.append(flux_pair)
                    flux_pair = []

                elif element.endswith(']'):
                    flux_pair.append(element[1:-2])
                    current_time.append(flux_pair)
                    time_steps.append(current_time)
                    current_time = []
                    flux_pair = []

            return time_steps

        result_file = path_ + '/results.xml'

        # make check for file
        if not os.path.isfile(result_file):
            raise FileNotFoundError('Cannot find results.xml in folder: ' + str(path_))

        # Initialize
        result_tree = ET.tostring(ET.parse(result_file).getroot())
        results = xmltodict.parse(result_tree)
        results_to_save = {}

        # find results
        for cell_ in results['result'].keys():
            results_to_save[cell_] = convert_to_list(str(results['result'][cell_]['surface_water_flux']))

        return results_to_save

    def process_flux(flux_dict_: dict, flux_config_: dict, center_points_: list) -> dict:

        # helper functions
        def get_wanted_fluxes(flux_tuple: list, center_points__) -> np.array:
            # delete entries where the flux is 0
            if flux_tuple[0] == 0:
                return np.array([])

            else:
                # delete entries that we don't want, if flux is not zero
                if flux_tuple[1].startswith('{Rain'):
                    if not flux_config_['rain']:
                        return np.array([])
                    else:
                        return np.array([0, 0, flux_tuple[0]])

                elif flux_tuple[1].startswith('{Evapo'):
                    if not flux_config_['evapotranspiration']:
                        return np.array([])
                    else:
                        return np.array([0, 0, flux_tuple[0]])

                elif flux_tuple[1].startswith('{Layer'):
                    if not flux_config_['infiltration']:
                        return np.array([])
                    else:
                        return np.array([0, 0, -flux_tuple[0]])

                elif flux_tuple[1].startswith('{Surface'):
                    if not flux_config_['run_off']:
                        return np.array([])
                    else:
                        if flux_tuple[0] > 0:
                            return np.array([])
                        else:
                            destination_cell = int(flux_tuple[1].split('#')[1][:-1])
                            local_cell = int(cell_.split('_')[1])
                            vector_ = center_points__[destination_cell] - center_points__[local_cell]
                            normalized_vector = vector_ / np.linalg.norm(vector_)
                            return normalized_vector * abs(flux_tuple[0])

        cell_vectors = {}
        for cell_ in flux_dict_.keys():
            time_vectors = []
            for time_step in range(0, len(flux_dict_[cell_])):
                vectors = []

                for flux in flux_dict_[cell_][time_step]:
                    vector = get_wanted_fluxes(flux, center_points_)
                    if len(vector) > 0:
                        vectors.append(vector)

                # Compute average vector
                if not vectors:
                    # if no vectors return [0, 0, 0]
                    time_vectors.append([0, 0, 0])
                else:
                    time_vectors.append(list(np.average(vectors, axis=0)))

            # add time vectors to cell
            cell_vectors[cell_] = time_vectors

        return cell_vectors

    # load files
    flux_config, center_points = read_files(path)
    flux_results = get_flux_result(path)

    processed_vectors = process_flux(flux_results, flux_config, center_points)

    # write file
    result_obj = open(path + '/surface_flux_result.txt', 'w')
    for cell in processed_vectors.keys():
        line = ''
        for point in processed_vectors[cell]:
            line += ','.join([str(p) for p in point]) + '\t'
        result_obj.write(line + '\n')

    result_obj.close()

    return True
