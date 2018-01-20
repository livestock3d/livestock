__author__ = "Christian Kongsgaard"
__license__ = "MIT"
__version__ = "0.0.1"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import numpy as np
import os
import multiprocessing

# Livestock imports


# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Air Functions

def new_temperature_and_relative_humidity(folder: str) -> bool:
    """
    Calculates a new temperatures and relative humidities for air volumes.
    :param folder: Path to folder containing case files.
    :return: True
    """

    # Helper functions
    def get_files(folder_: str) -> tuple:
        """
        Reads the case files and returns them as a tuple of lists
        :param folder_: Path to case folder
        :return: Tuple of lists
        """

        # helper functions
        def file_to_numpy(folder_path, file_name):
            file_obj = open(folder_path + '/' + file_name + '.txt', 'r')
            lines = file_obj.readlines()
            file_obj.close()
            clean_lines = [float(element.strip())
                           for line in lines
                           for element in line.split(',')]

            return np.array(clean_lines)

        def get_heights(folder_path):
            file_obj = open(folder_path + '/heights.txt', 'r')
            lines = file_obj.readlines()
            file_obj.close()

            return float(lines[0].strip()), float(lines[1].strip())

        def file_to_numpy_matrix(folder_path, file_name):
            file_obj = open(folder_path + '/' + file_name + '.txt', 'r')
            lines = file_obj.readlines()
            file_obj.close()
            clean_lines = [[float(element.strip())
                            for element in line.split(',')]
                           for line in lines]

            return np.array(clean_lines)

        def get_cpu(folder_path):
            file_obj = open(folder_path + '/cpu.txt', 'r')
            line = file_obj.readline()
            file_obj.close()

            return int(line.strip())

        # run function
        air_temperature_ = file_to_numpy(folder_, 'temperature')
        air_relative_humidity_ = file_to_numpy(folder_, 'relative_humidity')
        area_ = file_to_numpy(folder_, 'area')
        height_top_, height_stratification_ = get_heights(folder_)
        heat_flux_ = file_to_numpy_matrix(folder_, 'heat_flux')
        vapour_flux_ = file_to_numpy_matrix(folder_, 'vapour_flux')
        cpu_ = get_cpu(folder_)

        return air_temperature_, air_relative_humidity_, area_, height_top_, height_stratification_, heat_flux_, \
            vapour_flux_, cpu_

    def reconstruct_results(folder_, processed_rows_):
        # Sort row list
        sorted_rows = sorted(processed_rows_)

        # open result files
        temperature_file = open(folder_ + '/temperature_results.txt', 'w')
        relhum_file = open(folder_ + '/relative_humidity_results.txt', 'w')

        for row_number in sorted_rows:

            # process temperature files
            temp_path = folder_ + '/temp_' + str(row_number) + '.txt'
            temp_obj = open(temp_path, 'r')
            line = temp_obj.readline()
            temperature_file.write(line + '\n')
            temp_obj.close()
            os.remove(temp_path)

            # process relative humidity files
            relhum_path = folder + '/relhum_' + str(row_number) + '.txt'
            relhum_obj = open(relhum_path, 'r')
            line = relhum_obj.readline()
            relhum_file.write(line + '\n')
            relhum_obj.close()
            os.remove(relhum_path)

        temperature_file.close()
        relhum_file.close()

        return True

    # Run function
    temperature, relative_humidity, area, height_top, height_stratification, heat_flux, vapour_flux, cpu = \
        get_files(folder)

    rows_ = [i
             for i in range(0, len(heat_flux)-1)]

    input_packages = [(folder, index, temperature[index], relative_humidity[index], heat_flux[index],
                       vapour_flux[index], area, height_stratification, height_top)
                      for index in rows_]

    pool = multiprocessing.Pool(processes=cpu)
    processed_rows = pool.map(run_row, input_packages)
    reconstruct_results(folder, processed_rows)

    return True


def run_row(input_package: list) -> float:
    """
    Calculates a new temperatures and relative humidities for a row. A row represent all cell to a given time.

    :param input_package: Input package with need inputs.
    :type input_package: list
    :return: The row on which the calculation was performed.
    :rtype: float
    """

    # unpack
    folder_, row_index, temperature_time, relative_humidity_time, heat_flux_time_row, vapour_flux_time_row, area, \
        height_stratification, height_top = input_package

    # new mean temperature i K
    air_temperature_in_k = celsius_to_kelvin(temperature_time)
    temperature_row = new_mean_temperature(area,
                                           height_top,
                                           air_temperature_in_k,
                                           heat_flux_time_row)

    # air flow
    air_flow_row = air_flow(area,
                            height_top,
                            air_temperature_in_k,
                            temperature_row)

    # new relative humidity
    relative_humidity_row = new_mean_relative_humidity(area,
                                                       height_top,
                                                       temperature_row,
                                                       relative_humidity_to_vapour_pressure(
                                                           relative_humidity_time,
                                                           air_temperature_in_k),
                                                       vapour_flux_time_row,
                                                       air_flow_row
                                                       )

    # new stratified relative humidity
    stratified_relative_humidity_row = stratification(height_stratification,
                                                      relative_humidity_row,
                                                      height_top,
                                                      relative_humidity_time
                                                      )

    # new stratified temperature in C
    stratified_temperature_row = stratification(height_stratification,
                                                kelvin_to_celsius(temperature_row),
                                                height_top,
                                                temperature_time)

    # write results
    temp_file = open(folder_ + '/temp_' + str(row_index) + '.txt', 'w')
    temp_file.write(','.join(stratified_temperature_row.astype(str)))
    temp_file.close()

    relhum_file = open(folder_ + '/relhum_' + str(row_index) + '.txt', 'w')
    relhum_file.write(','.join(stratified_relative_humidity_row.astype(str)))
    relhum_file.close()

    return row_index


def new_mean_relative_humidity(area, height_external, temperature_internal, vapour_pressure_external,
                               vapour_production, air_flow_):

    vapour_pressure = new_mean_vapour_pressure(area, height_external, temperature_internal, vapour_pressure_external,
                                               vapour_production, air_flow_)

    return vapour_pressure_to_relative_humidity(vapour_pressure, temperature_internal)


def new_mean_vapour_pressure(area, height_external, temperature_internal, vapour_pressure_external,
                             vapour_production, air_flow_):
    """
    Calculates a new vapour pressure for the volume.
    :param area: area in m^2
    :param temperature_internal: external temperature in K
    :param height_external: external height in m
    :param vapour_pressure_external: external vapour pressure in Pa
    :param vapour_production: vapour production in kg/s
    :param air_flow_: air flow in m^3/s
    :return: new vapour pressure in Pa
    """

    volume_air = area * height_external  # m^3
    gas_constant_vapour = 461.5  # J/kgK

    contact = air_flow_ / (gas_constant_vapour * temperature_internal)  # kg/sPa
    capacity = volume_air / (gas_constant_vapour * temperature_internal)  # m3kg/J
    vapour_pressure = vapour_pressure_external + vapour_production/contact * (1 - np.exp(-contact/capacity))  # Pa

    return vapour_pressure


def air_flow(area, height_top, temperature_top, temperature_mean):
    """
    Calculates an air flow based on an mean temperature for the volume.
    :param area: in m^2
    :param height_top: in m
    :param temperature_top: in K
    :param temperature_mean: in K
    :return: air flow in m^3/s
    """

    density_air = 1.29 * 273 / temperature_top  # kg/m^3
    gravity = 9.81  # m/s^2
    height_mean = height_top / 2  # m

    delta_temperature = temperature_top - temperature_mean
    delta_pressure = density_air * gravity * (height_top - height_mean) * delta_temperature / temperature_mean

    return area * np.sqrt(2 * abs(delta_pressure) / density_air) * delta_pressure / abs(delta_pressure)


def new_mean_temperature(area, height_external, temperature_external, heat):
    """
    Calculates a new mean temperature for the volume.
    :param area: in m^2
    :param height_external: in m
    :param temperature_external: in K
    :param heat: in J
    :return: temperature in K
    """

    volume_air = area * height_external
    specific_heat_capacity = 1005  # J/kgK
    density_air = 1.29 * 273 / temperature_external  # kg/m^3
    energy_air = volume_air * specific_heat_capacity * density_air * temperature_external  # J

    return (energy_air + heat)/(volume_air * density_air * specific_heat_capacity)


def celsius_to_kelvin(celsius):
    kelvin = celsius + 273
    return kelvin


def kelvin_to_celsius(kelvin):
    celsius = kelvin - 273
    return celsius


def vapour_pressure_to_relative_humidity(vapour_pressure, temperature):
    """
    Convert vapour pressure to relative humidity
    :param vapour_pressure: in Pa
    :param temperature: in K
    :return: relative humidity as unitless
    """

    temperature_c = kelvin_to_celsius(temperature)  # C
    saturated_pressure = 288.68 * (1.098 + temperature_c/100)**8.02  # Pa
    relative_humidity = vapour_pressure/saturated_pressure  # -

    return relative_humidity


def relative_humidity_to_vapour_pressure(relative_humidity: float, temperature: float) -> float:
    """
    Convert relative humidity to vapour pressure.

    :type temperature: float
    :type relative_humidity: float
    :param relative_humidity: unitless
    :param temperature: in K
    :return: vapour pressure in Pa
    """

    temperature_c = kelvin_to_celsius(temperature)  # C
    saturated_pressure = 288.68 * (1.098 + temperature_c/100)**8.02  # Pa
    vapour_pressure = relative_humidity * saturated_pressure  # Pa

    return vapour_pressure


def stratification(height: float, value_mean: float, height_top: float, value_top: float):
    """
    Calculates the stratification of the temperature or relative humidity

    :param height: height at which the stratification value is wanted. in m.
    :param value_mean: mean value
    :param height_top: height at the top of the boundary. in m
    :param value_top: value at the top of the boundary
    :return: value at desired height.
    """

    return value_mean - 2 * height * (value_mean - value_top)/height_top


def water_evaporation(volume_air: float, temperature: float, water: float, fraction_of_evaporation: float) -> tuple:
    """

    :param volume_air:
    :type volume_air:
    :param temperature:
    :type temperature:
    :param water:
    :type water:
    :param fraction_of_evaporation:
    :type fraction_of_evaporation:
    :return:
    :rtype:
    """

    specific_heat_capacity = 1005  # J/kgK
    density_air = 1.29 * 273 / temperature  # kg/m^3
    heat_of_evaporation_water = 2257000  # J/kg
    density_water = 1000  # kg/m3

    vapour_gain = water/density_water * fraction_of_evaporation
    energy_of_evaporation = water * heat_of_evaporation_water * fraction_of_evaporation
    energy_air = volume_air * specific_heat_capacity * density_air * temperature  # J
    new_temperature = (energy_air - energy_of_evaporation)/(volume_air * specific_heat_capacity * density_air)

    return new_temperature, energy_of_evaporation, vapour_gain

def water_evaporation_template_wrapper():

    def read_files(result_path):

        volume_file = open(result_path + '/volume.txt', 'r')
        volume_lines = volume_file.readlines()
        volume_file.close()
        volume_ = [float(v.strip())
                  for line in volume_lines
                  for v in line.split(',')]

        # Create Temperature file
        temp_file = open(result_path + '/temperature.txt', 'r')
        temp_lines = temp_file.readlines()
        temp_file.close()
        temperature_ = [float(t.strip())
                       for line in temp_lines
                       for t in line.split(',')]

        # Create water file
        water_file = open(result_path + '/water.txt', 'r')
        water_lines = water_file.readlines()
        water_file.close()
        water_ = [float(w.strip())
                 for line in water_lines
                 for w in line.split(',')]

        # Create fraction file
        fraction_file = open(result_path + '/fraction.txt', 'r')
        fraction_lines = fraction_file.readlines()
        fraction_file.close()
        fraction_ = [float(f.strip())
                    for line in fraction_lines
                    for f in line.split(',')]

        return volume_, temperature_, water_, fraction_

    def write_files(write_path, temperature, energy, vapour):

        # Create Temperature file
        temp_file = open(write_path + '/temperature.txt', 'w')
        temp_file.write(','.join([str(t)
                                  for t in temperature]))
        temp_file.close()

        # Create volume file
        energy_file = open(write_path + '/energy.txt', 'w')
        energy_file.write(','.join([str(e)
                                    for e in energy]))
        energy_file.close()

        # Create water file
        vapour_file = open(write_path + '/vapour.txt', 'w')
        vapour_file.write(','.join([str(v)
                                   for v in vapour]))
        vapour_file.close()

        return True

    local_path = r'C:\livestock\local'
    volume, temperature, water, fraction = read_files(local_path)
    new_temperature = []
    energy_of_evaporation = []
    vapour_gain = []

    for i in range(0, len(volume)):
        nt, ee, vg = water_evaporation(volume[i], temperature[i], water[i], fraction[i])
        new_temperature.append(nt)
        energy_of_evaporation.append(ee)
        vapour_gain.append(vg)

    write_files(local_path, new_temperature, energy_of_evaporation, vapour_gain)