__author__ = "Christian Kongsgaard"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import numpy as np
import multiprocessing
from scipy.optimize import brentq

# Livestock imports


# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Air Functions

def new_temperature_and_relative_humidity(folder: str) -> bool:
    """
    Calculates a new temperatures and relative humidity for air volumes.

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
        wind_speed_ = np.loadtxt(folder_ + '/wind_speed.txt', delimiter=',')
        area_ = np.loadtxt(folder_ + '/area.txt', delimiter=',')
        height_top_, height_stratification_ = np.loadtxt(folder_ + '/heights.txt', delimiter=',')
        vapour_flux_ = np.loadtxt(folder_ + '/vapour_flux.txt', delimiter=',')
        cpu_ = np.loadtxt(folder + '/cpu.txt', delimiter=',')

        return air_temperature_, air_relative_humidity_, wind_speed_, area_, height_top_, height_stratification_, \
            vapour_flux_, int(cpu_)

    def reconstruct_results(folder_, processed_rows_):

        # Sort row list
        sorted_rows = sorted(processed_rows_)

        # create lists
        temperature_ = []
        relative_humidity_ = []
        latent_heat_flux_ = []
        vapour_flux_ = []

        for row in sorted_rows:
            temperature_.append(row[1])
            relative_humidity_.append(row[2])
            latent_heat_flux_.append(row[3])
            vapour_flux_.append(row[4])

        np.savetxt(folder_ + '/temperature_results.txt', temperature_, delimiter=',', fmt='%.4f')
        np.savetxt(folder_ + '/relative_humidity_results.txt', relative_humidity_, delimiter=',', fmt='%.4f')
        np.savetxt(folder_ + '/latent_heat_flux_results.txt', latent_heat_flux_, delimiter=',', fmt='%.4f')
        np.savetxt(folder_ + '/vapour_flux_results.txt', vapour_flux_, delimiter=',', fmt='%.4f')

        return True

    # Run function
    temperature, relative_humidity, wind_speed, area, height_top, height_stratification, vapour_flux, cpu = \
        get_files(folder)

    rows_ = [i
             for i in range(0, len(vapour_flux)-1)]

    input_packages = [(index,
                       temperature[index],
                       convert_relative_humidity_to_unitless(relative_humidity[index]),
                       vapour_flux[index],
                       wind_speed[index],
                       area,
                       height_stratification,
                       height_top)
                      for index in rows_]

    pool = multiprocessing.Pool(processes=cpu)
    processed_rows = pool.map(run_row, input_packages)
    reconstruct_results(folder, processed_rows)

    return True


def max_possible_vapour_flux(vapour_mass_flux, volume, temperature_in_kelvin, vapour_pressure):

    # Fsolve function
    new_temperature = new_mean_temperature(volume,
                                           temperature_in_kelvin,
                                           latent_heat_flux(vapour_mass_flux))

    saturated_pressure = saturated_vapour_pressure(new_temperature)
    actual_pressure = new_mean_vapour_pressure(volume, new_temperature, vapour_pressure, vapour_mass_flux)
    return saturated_pressure - actual_pressure


def compute_temperature_relative_humidity(temperature_in_k, relative_humidity, vapour_mass_flux, volume):

    # Check if all cells have a vapour pressure below saturation after alteration of temp & relhum
    new_temperature_check = new_mean_temperature(volume, temperature_in_k,
                                                 latent_heat_flux(vapour_mass_flux))

    vapour_pressure = relative_humidity_to_vapour_pressure(relative_humidity,
                                                           temperature_in_k)

    available_vapour_mass_flux = []
    for i in range(0, len(vapour_mass_flux)):
        new_vapour_pressure = new_mean_vapour_pressure(volume[i],
                                                       new_temperature_check[i],
                                                       vapour_pressure,
                                                       vapour_mass_flux[i])

        new_saturated_pressure = saturated_vapour_pressure(new_temperature_check[i])

        if new_vapour_pressure <= new_saturated_pressure:
            available_vapour_mass_flux.append(vapour_mass_flux[i])
        else:
            possible_vapour_mass_flux = brentq(max_possible_vapour_flux,
                                               -1,
                                               vapour_mass_flux[i],
                                               args=(volume[i], temperature_in_k, vapour_pressure))

            if possible_vapour_mass_flux < vapour_mass_flux[i]:
                available_vapour_mass_flux.append(possible_vapour_mass_flux)
            else:
                # Should be a useless statement?
                available_vapour_mass_flux.append(vapour_mass_flux[i])

    available_vapour_mass_flux = np.array(available_vapour_mass_flux).flatten()
    latent_heat = latent_heat_flux(available_vapour_mass_flux)
    new_temperature_ = new_mean_temperature(volume, temperature_in_k, latent_heat)
    new_relative_humidity = new_mean_relative_humidity(volume,
                                                       new_temperature_,
                                                       vapour_pressure,
                                                       available_vapour_mass_flux)
    return (new_temperature_,
            new_relative_humidity,
            latent_heat,
            available_vapour_mass_flux)


def run_row(input_package: list) -> tuple:
    """
    Calculates a new temperatures and relative humidity for a row. A row represent all cells to a given time.

    :param input_package: Input package with need inputs.
    :type input_package: list
    :return: The row on which the calculation was performed.
    :rtype: tuple
    """

    # unpack
    row_index, temperature_time, relative_humidity_time, vapour_flux_time_row, wind_speed, area, \
        height_stratification, height_top = input_package

    # new mean temperature i K
    air_temperature_in_k = celsius_to_kelvin(temperature_time)
    #flux_corrected_volume = area * height_top + wind_speed_to_flux(wind_speed, height_top, cross_section_from_area(area))
    # Get temperature and relative_humidity
    [temperature_row,
     relative_humidity_row,
     heat_flux_time_row,
     used_vapour_flux] = compute_temperature_relative_humidity(air_temperature_in_k,
                                                               relative_humidity_time,
                                                               convert_vapour_flux_to_kgh(vapour_flux_time_row),
                                                               height_top * area)

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

    return (row_index,
            stratified_temperature_row,
            convert_relative_humidity_to_percentage(stratified_relative_humidity_row),
            heat_flux_time_row,
            used_vapour_flux)


def latent_heat_flux(vapour_mass_flux: np.array) -> np.array:
    """
    Computes the latent heat flux related to a certain evapotranspiration flux.
    The latent heat flux is negative if the vapour flux is positive.

    Source: Manickathan, L. et al., 2018.
    Parametric study of the influence of environmental factors and tree properties on the
    transpirative cooling effect of trees. Agricultural and Forest Meteorology.

    :param vapour_mass_flux: Vapour volume flux in kg/h
    :type vapour_mass_flux: numpy.array
    :return: Latent heat flux in J/h.
    :rtype: numpy.array
    """

    latent_heat_of_vaporization = 2.5 * 10**6  # J/kg

    return np.negative(latent_heat_of_vaporization * vapour_mass_flux)  # J/h


def convert_vapour_flux_to_kgh(vapour_flux: np.array) -> np.array:
    """
    Converts a vapour flux from m\ :sup:`3`\/day to kg/h
    Density of water: 1000kg/m\ :sup:`3`\
    Hours per day: 24h/day
    Conversion: 1000kg/m\ :sup:`3`\ / 24h/day

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


def new_mean_relative_humidity(volume: np.array, temperature_internal: np.array,
                               vapour_pressure_external: np.array, vapour_production: np.array) -> np.array:
    """
    Computes a new mean vapour pressure and converts it in to a relative humidity.

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param volume: Air volume in m\ :sup:`3`
    :type volume: numpy.array
    :param temperature_internal: External temperature in K
    :type temperature_internal: numpy.array
    :param vapour_pressure_external: External vapour pressure in Pa
    :type vapour_pressure_external: numpy.array
    :param vapour_production: Vapour production in kg/h
    :type vapour_production: numpy.array
    :return: Relative humidity - unitless
    :rtype: numpy.array
    """

    vapour_pressure = new_mean_vapour_pressure(volume, temperature_internal, vapour_pressure_external,
                                               vapour_production)

    return vapour_pressure_to_relative_humidity(vapour_pressure, temperature_internal)


def new_mean_vapour_pressure(volume: np.array, temperature: np.array,
                             vapour_pressure_external: np.array, vapour_production: np.array) -> np.array:
    """
    Calculates a new vapour pressure for the volume.

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param volume: Volume in m\ :sup:`3`
    :type volume: numpy.array
    :param temperature: Temperature in K
    :type temperature: numpy.array
    :param vapour_pressure_external: External vapour pressure in Pa
    :type vapour_pressure_external: numpy.array
    :param vapour_production: Vapour production in kg/h
    :type vapour_production: numpy.array
    :return: New vapour pressure in Pa
    :rtype: numpy.array
    """

    gas_constant_vapour = 461.5  # J/kgK
    density_vapour = vapour_pressure_external/(gas_constant_vapour * temperature)  # kg/m3
    mass_vapour = density_vapour * volume  # kg
    new_mass_vapour = mass_vapour + vapour_production  # kg
    vapour_pressure = (new_mass_vapour * gas_constant_vapour * temperature) / volume  # Pa

    return vapour_pressure  # Pa


def new_mean_temperature(volume: np.array, temperature: np.array, heat: np.array) -> np.array:
    """
    Calculates a new mean temperature for the volume.

    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param volume: Volume in m\ :sup:`3`
    :type volume: numpy.array
    :param temperature: Temperature at the top of the air volume in K
    :type temperature: numpy.array
    :param heat: Added heat to the air volume in J/h
    :type heat: numpy.array
    :return: Temperature in K
    :rtype: numpy.array
    """

    specific_heat_capacity = 1005  # J/kgK
    density_air = 1.29 * 273.15 / temperature  # kg/m^3
    energy_air = volume * specific_heat_capacity * density_air * temperature  # J

    return (energy_air + heat)/(volume * density_air * specific_heat_capacity)


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

    return vapour_pressure / saturated_vapour_pressure(temperature)  # -


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

    return relative_humidity * saturated_vapour_pressure(temperature)  # Pa


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


def saturated_vapour_pressure(temperature: float) -> float:
    """
    Computes the saturated vapour pressure for a given temperature.
    Source: Peuhkuri, Ruut, and Carsten Rode. 2016.
    “Heat and Mass Transfer in Buildings.”

    :param temperature: Temperature in Kelvin
    :type temperature: float
    :return: Vapour pressure in Pa
    :rtype: float
    """

    # Make check. -109.8C is the temperature where the air can hold no water.
    temperature_in_celsius = np.maximum(kelvin_to_celsius(temperature), -109)
    return 288.68 * (1.098 + temperature_in_celsius/100) ** 8.02


def wind_speed_to_hour_flux(wind_speed: float) -> float:
    """
    Converts wind speed into a hourly flux.
    m/s to m\ :sup:`3`/h
    m/s to m\ :sup:`3`/s = 1:sup:`2`
    m\ :sup:`3`/s to m\ :sup:`3`/h = 3600s/h

    :param wind_speed: Wind speed in m/s
    :type wind_speed: float
    :return: Wind flux in m\ :sup:`3`/h
    :rtype: float
    """

    return wind_speed * 3600


def cross_section_from_area(area: np.array) -> np.array:

    return np.sqrt(4 * area/np.pi)


def wind_speed_to_flux(wind_speed, height, cross_section):

    return wind_speed_to_hour_flux(wind_speed) * height * cross_section
