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
    :type folder: str
    :return: True
    :rtype: bool
    """

    # Helper functions
    def get_files(folder_: str) -> tuple:
        """
        Reads the case files and returns them as a tuple of lists
        :param folder_: Path to case folder
        :return: Tuple of lists
        """

        # Load
        air_temperature_ = np.loadtxt(folder_ + '/temperature.txt', delimiter=',')
        air_relative_humidity_ = np.loadtxt(folder_ + '/relative_humidity.txt', delimiter=',')
        area_ = np.loadtxt(folder_ + '/area.txt', delimiter=',')
        height_top_, height_stratification_ = np.loadtxt(folder_ + '/heights.txt', delimiter=',')
        vapour_flux_ = np.loadtxt(folder_ + '/vapour_flux.txt', delimiter=',')
        cpu_ = np.loadtxt(folder + '/cpu.txt', delimiter=',')

        return air_temperature_, air_relative_humidity_, area_, height_top_, height_stratification_, \
            vapour_flux_, int(cpu_)

    def reconstruct_results(folder_, processed_rows_):

        # Sort row list
        sorted_rows = sorted(processed_rows_)

        # create lists
        temperature_ = []
        relative_humidity_ = []
        latent_heat_flux_ = []

        for row in sorted_rows:
            temperature_.append(row[1])
            relative_humidity_.append(row[2])
            latent_heat_flux_.append(row[3])

        np.savetxt(folder_ + '/temperature_results.txt', temperature_, delimiter=',', fmt='%.4f')
        np.savetxt(folder_ + '/relative_humidity_results.txt', relative_humidity_, delimiter=',', fmt='%.4f')
        np.savetxt(folder_ + '/latent_heat_flux_results.txt', latent_heat_flux_, delimiter=',', fmt='%.4f')

        return True

    # Run function
    temperature, relative_humidity, area, height_top, height_stratification, vapour_flux, cpu = \
        get_files(folder)

    rows_ = [i
             for i in range(0, len(vapour_flux)-1)]

    input_packages = [(index, temperature[index], convert_relative_humidity_to_unitless(relative_humidity[index]),
                       latent_heat_flux(vapour_flux[index]), convert_vapour_flux_to_kgs(vapour_flux[index]),
                       area, height_stratification, height_top)
                      for index in rows_]

    pool = multiprocessing.Pool(processes=cpu)
    processed_rows = pool.map(run_row, input_packages)
    reconstruct_results(folder, processed_rows)

    return True


def run_row(input_package: list) -> tuple:
    """
    Calculates a new temperatures and relative humidities for a row. A row represent all cells to a given time.

    :param input_package: Input package with need inputs.
    :type input_package: list
    :return: The row on which the calculation was performed.
    :rtype: tuple
    """

    # unpack
    row_index, temperature_time, relative_humidity_time, heat_flux_time_row, vapour_flux_time_row, area, \
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


    return (row_index, stratified_temperature_row,
            convert_relative_humidity_to_percentage(stratified_relative_humidity_row), heat_flux_time_row)


def latent_heat_flux(vapour_volume_flux: np.array) -> np.array:
    """
    Computes the latent heat flux related to a certain evapotranspiration flux.
    Source: Manickathan, L. et al., 2018.
    Parametric study of the influence of environmental factors and tree properties on the
    transpirative cooling effect of trees. Agricultural and Forest Meteorology.

    :param vapour_volume_flux: Vapour volume flux in m\ :sup:`3`\/day
    :type vapour_volume_flux: numpy.array
    :return: latent heat flux in J/h
    :rtype: numpy.array
    """

    latent_heat_of_vaporization = 2.5 * 10**6  # J/kg
    vapour_mass_flux = vapour_volume_flux * 1000/24  # kg/h

    return latent_heat_of_vaporization * vapour_mass_flux  # J/h


def convert_vapour_flux_to_kgs(vapour_flux: np.array) -> np.array:
    """
    Converts a vapour flux from m\ :sup:`3`\/day to kg/s
    Density of water: 1000kg/m\ :sup:`3`\
    Seconds per day: 24h * 60min * 60s = 86400s/day
    Conversion: 1000kg/m\ :sup:`3`\/86400s/day

    :param vapour_flux: Vapour flux in m\ :sup:`3`\/day
    :type vapour_flux: numpy.array
    :return: Vapour flux in kg/s
    :rtype: numpy.array
    """

    return vapour_flux * 1000 / 86400


def convert_relative_humidity_to_unitless(rh: np.array) -> np.array:
    """
    Converts relative humidity from percentage to an unit less number.

    :param rh: Relative humidity in %
    :type rh: numpy.array
    :return: Relative humidity as unitless
    :rtype: numpy.array
    """

    return rh / 100


def convert_relative_humidity_to_percentage(rh: np.array) -> np.array:
    """
    Converts relative humidity from percentage to an unit less number.

    :param rh: Relative humidity as unitless
    :type rh: numpy.array
    :return: Relative humidity in percentage
    :rtype: numpy.array
    """

    return rh * 100


def new_mean_relative_humidity(area: np.array, height_external: float, temperature_internal: np.array,
                               vapour_pressure_external: np.array, vapour_production: np.array,
                               air_flow_: np.array) -> np.array:
    """
    Computes a new mean vapour pressure and converts it in to a relative humidity.

    Source: ????

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param height_external: External height in m
    :type height_external: numpy.array
    :param temperature_internal: External temperature in K
    :type temperature_internal: numpy.array
    :param vapour_pressure_external: External vapour pressure in Pa
    :type vapour_pressure_external: numpy.array
    :param vapour_production: Vapour production in kg/s
    :type vapour_production: numpy.array
    :param air_flow_: Air flow in m\ :sup:`3`\/s
    :type air_flow_: numpy.array
    :return: Relative humidity - unitless
    :rtype: numpy.array
    """

    vapour_pressure = new_mean_vapour_pressure(area, height_external, temperature_internal, vapour_pressure_external,
                                               vapour_production, air_flow_)

    return vapour_pressure_to_relative_humidity(vapour_pressure, temperature_internal)


def new_mean_vapour_pressure(area: np.array, height_external: float, temperature_internal: np.array,
                             vapour_pressure_external: np.array, vapour_production: np.array,
                             air_flow_: np.array) -> np.array:
    """
    Calculates a new vapour pressure for the volume.
    Source: ????

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param temperature_internal: External temperature in K
    :type temperature_internal: numpy.array
    :param height_external: External height in m
    :type height_external: float
    :param vapour_pressure_external: External vapour pressure in Pa
    :type vapour_pressure_external: numpy.array
    :param vapour_production: Vapour production in kg/s
    :type vapour_production: numpy.array
    :param air_flow_: Air flow in m\ :sup:`3`\/s
    :type air_flow_: numpy.array
    :return: New vapour pressure in Pa
    :rtype: numpy.array
    """

    volume_air = area * height_external  # m^3
    gas_constant_vapour = 461.5  # J/kgK

    contact = air_flow_ / (gas_constant_vapour * temperature_internal)  # kg/sPa
    capacity = volume_air / (gas_constant_vapour * temperature_internal)  # m3kg/J
    vapour_pressure = vapour_pressure_external + vapour_production/contact * (1 - np.exp(-contact/capacity))  # Pa

    return vapour_pressure


def air_flow(area: np.array, height_top: float, temperature_top: np.array, temperature_mean: np.array) -> np.array:
    """
    Calculates an air flow based on an mean temperature for the volume.

    Source: ???

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param height_top: Top of the air volume in m
    :type height_top: float
    :param temperature_top: Temperature at the top of the air volume in K
    :type temperature_top: numpy.array
    :param temperature_mean: Mean Temperature of the volume in K
    :type temperature_mean: numpy.array
    :return: Air flow in m\ :sup:`3`\/s
    :rtype: numpy.array
    """

    density_air = 1.29 * 273 / temperature_top  # kg/m^3
    gravity = 9.81  # m/s^2
    height_mean = height_top / 2  # m

    delta_temperature = temperature_top - temperature_mean
    delta_pressure = density_air * gravity * (height_top - height_mean) * delta_temperature / temperature_mean

    return area * np.sqrt(2 * np.absolute(delta_pressure) / density_air) * delta_pressure / np.absolute(delta_pressure)


def new_mean_temperature(area: np.array, height_external: float, temperature_external: np.array, heat: np.array) -> np.array:
    """
    Calculates a new mean temperature for the volume.

    Source: ???

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param height_external: Top of the air volume in m
    :type height_external: float
    :param temperature_external: Temperature at the top of the air volume in K
    :type temperature_external: numpy.array
    :param heat: Added heat to the air volume in J
    :type heat: numpy.array
    :return: Temperature in K
    :rtype: numpy.array
    """

    volume_air = area * height_external
    specific_heat_capacity = 1005  # J/kgK
    density_air = 1.29 * 273 / temperature_external  # kg/m^3
    energy_air = volume_air * specific_heat_capacity * density_air * temperature_external  # J

    return (energy_air + heat)/(volume_air * density_air * specific_heat_capacity)


def celsius_to_kelvin(celsius: float) -> float:
    """
    Converts a temperature in Celsius to Kelvin.

    Source: https://en.wikipedia.org/wiki/Celsius

    :param celsius: Temperature in Celsius
    :type celsius: float
    :return: Temperature in Kelvin
    :rtype: float
    """

    kelvin = celsius + 273.15

    return kelvin


def kelvin_to_celsius(kelvin: float) -> float:
    """
    Converts a temperature in Kelvin to Celsius.

    Source: https://en.wikipedia.org/wiki/Celsius

    :param kelvin: Temperature in Kelvin
    :type kelvin: float
    :return: Temperature in Celsius
    :rtype: float
    """

    celsius = kelvin - 273.15
    return celsius


def vapour_pressure_to_relative_humidity(vapour_pressure: float, temperature: float) -> float:
    """
    Convert vapour pressure to relative humidity given a air temperature

    Source: ???

    :param vapour_pressure: Vapour pressure in Pa
    :type vapour_pressure: float
    :param temperature: Air temperature in K
    :type temperature: float
    :return: Relative humidity as unitless
    :rtype: float
    """

    temperature_c = kelvin_to_celsius(temperature)  # C
    saturated_pressure = 288.68 * (1.098 + temperature_c/100)**8.02  # Pa
    relative_humidity = vapour_pressure/saturated_pressure  # -

    return relative_humidity


def relative_humidity_to_vapour_pressure(relative_humidity: float, temperature: float) -> float:
    """
    Convert relative humidity to vapour pressure given a air temperature.

    Source: ???

    :type temperature: float
    :type relative_humidity: float
    :param relative_humidity: Relative humidity - unitless
    :param temperature: Air temperature in K
    :return: Vapour pressure in Pa
    :rtype: float
    """

    temperature_c = kelvin_to_celsius(temperature)  # C
    saturated_pressure = 288.68 * (1.098 + temperature_c/100)**8.02  # Pa
    vapour_pressure = relative_humidity * saturated_pressure  # Pa

    return vapour_pressure


def stratification(height: float, value_mean: float, height_top: float, value_top: float) -> float:
    """
    Calculates the stratification of the temperature or relative humidity of the air volume.

    Source: ???

    :param height: Height at which the stratification value is wanted. in m.
    :type height: float
    :param value_mean: Mean value of the air volume. Assumed equal to the value at half of the height of the air volume.
    :type value_mean: float
    :param height_top: Height at the top of the boundary. in m
    :type height_top: float
    :param value_top: Value at the top of the air volume
    :type value_top: float
    :return: Value at desired height.
    :rtype: float
    """

    return value_mean - 2 * height * (value_mean - value_top)/height_top


def water_evaporation(volume_air: float, temperature: float, water: float, fraction_of_evaporation: float) -> tuple:
    """
    Computes the change in temperature for a volume caused by the evaporation of a given water volume.

    :param volume_air: Air volume in m3
    :type volume_air: float
    :param temperature: Air temperature in K
    :type temperature: float
    :param water: Water introduced into the air volume to lower the temperature in m3
    :type water: float
    :param fraction_of_evaporation: Fraction of the water which is evaporated.
    :type fraction_of_evaporation: float
    :return: New temperature in K, energy needed for the evaporation in J and the vapour gain of the volume in kg
    :rtype: float
    """

    specific_heat_capacity = 1005  # J/kgK
    density_air = 1.29 * 273 / temperature  # kg/m^3
    heat_of_evaporation_water = 2257000  # J/kg
    density_water = 1000  # kg/m3

    vapour_gain = water * density_water * fraction_of_evaporation  # kg
    energy_of_evaporation = water * heat_of_evaporation_water * fraction_of_evaporation  # J
    energy_air = volume_air * specific_heat_capacity * density_air * temperature  # J
    new_temperature = (energy_air - energy_of_evaporation)/(volume_air * specific_heat_capacity * density_air)  #K

    return new_temperature, energy_of_evaporation, vapour_gain


def water_evaporation_template_wrapper(local_path):

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

    def write_files(write_path, temperature_, energy, vapour):

        # Create Temperature file
        temp_file = open(write_path + '/temperature.txt', 'w')
        temp_file.write(','.join([str(t)
                                  for t in temperature_]))
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
