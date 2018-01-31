__author__ = "Christian Kongsgaard"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import numpy as np
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
                       latent_heat_flux(vapour_flux[index]), convert_vapour_flux_to_kgh(vapour_flux[index]),
                       area, height_stratification, height_top)
                      for index in rows_]

    pool = multiprocessing.Pool(processes=cpu)
    processed_rows = pool.map(run_row, input_packages)
    reconstruct_results(folder, processed_rows)

    return True


def run_row(input_package: list) -> tuple:
    """
    Calculates a new temperatures and relative humidity for a row. A row represent all cells to a given time.

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

    # new relative humidity
    relative_humidity_row = new_mean_relative_humidity(area,
                                                       height_top,
                                                       temperature_row,
                                                       relative_humidity_to_vapour_pressure(
                                                           relative_humidity_time,
                                                           air_temperature_in_k),
                                                       vapour_flux_time_row,
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


def convert_vapour_flux_to_kgh(vapour_flux: np.array) -> np.array:
    """
    Converts a vapour flux from m\ :sup:`3`\/day to kg/h
    Density of water: 1000kg/m\ :sup:`3`\
    Hours per day: 24h/day
    Conversion: 1000kg/m\ :sup:`3`\/24h/day

    :param vapour_flux: Vapour flux in m\ :sup:`3`\/day
    :type vapour_flux: numpy.array
    :return: Vapour flux in kg/h
    :rtype: numpy.array
    """

    return vapour_flux * 1000 / 24


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
                               vapour_pressure_external: np.array, vapour_production: np.array) -> np.array:
    """
    Computes a new mean vapour pressure and converts it in to a relative humidity.

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param height_external: External height in m
    :type height_external: numpy.array
    :param temperature_internal: External temperature in K
    :type temperature_internal: numpy.array
    :param vapour_pressure_external: External vapour pressure in Pa
    :type vapour_pressure_external: numpy.array
    :param vapour_production: Vapour production in kg/h
    :type vapour_production: numpy.array
    :return: Relative humidity - unitless
    :rtype: numpy.array
    """

    vapour_pressure = new_mean_vapour_pressure(area, height_external, temperature_internal, vapour_pressure_external,
                                               vapour_production)

    return vapour_pressure_to_relative_humidity(vapour_pressure, temperature_internal)


def new_mean_vapour_pressure(area: np.array, height_external: float, temperature_internal: np.array,
                             vapour_pressure_external: np.array, vapour_production: np.array) -> np.array:
    """
    Calculates a new vapour pressure for the volume.

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param temperature_internal: External temperature in K
    :type temperature_internal: numpy.array
    :param height_external: External height in m
    :type height_external: float
    :param vapour_pressure_external: External vapour pressure in Pa
    :type vapour_pressure_external: numpy.array
    :param vapour_production: Vapour production in kg/h
    :type vapour_production: numpy.array
    :return: New vapour pressure in Pa
    :rtype: numpy.array
    """

    volume_air = area * height_external  # m3
    gas_constant_vapour = 461.5  # J/kgK
    density_vapour = vapour_pressure_external/(gas_constant_vapour * temperature_internal)  # kg/m3
    mass_vapour = density_vapour * volume_air  # kg
    new_mass_vapour = mass_vapour + vapour_production  # kg
    new_density_vapour = new_mass_vapour / volume_air  # kg/m3
    vapour_pressure = new_density_vapour * gas_constant_vapour * temperature_internal # Pa

    return vapour_pressure  # Pa


def new_mean_temperature(area: np.array, height_external: float, temperature_external: np.array, heat: np.array) -> np.array:
    """
    Calculates a new mean temperature for the volume.

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param area: Area in m\ :sup:`2`
    :type area: numpy.array
    :param height_external: Top of the air volume in m
    :type height_external: float
    :param temperature_external: Temperature at the top of the air volume in K
    :type temperature_external: numpy.array
    :param heat: Added heat to the air volume in J/h
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

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

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

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

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

    :param height: Height at which the stratification value is wanted in m.
    :type height: float
    :param value_mean: Mean value of the air volume. Assumed equal to the value at half of the height of the air volume.
    :type value_mean: float
    :param height_top: Height at the top of the boundary in m.
    :type height_top: float
    :param value_top: Value at the top of the air volume
    :type value_top: float
    :return: Value at desired height.
    :rtype: float
    """

    return value_mean - 2 * height * (value_mean - value_top)/height_top
