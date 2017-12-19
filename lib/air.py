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

class NewTemperatureAndRelativeHumidity:

    def __init__(self, folder):
        self.folder = folder
        self.air_temperature = None
        self.air_relative_humidity = None
        self.area = None
        self.height_top = None
        self.height_stratification = None
        self.heat_flux = None
        self.vapour_flux = None
        self.processes = 3

    def get_files(self):

        def file_to_numpy(folder, file_name):
            file_obj = open(folder + '/' + file_name + '.txt', 'r')
            lines = np.array(file_obj.readlines())
            file_obj.close()

            def func(string):
                return float(string.strip('\n'))

            return np.apply_along_axis(func, 1, lines)

        def get_heights(folder):
            file_obj = open(folder + '/heights.txt', 'r')
            lines = file_obj.readlines()
            file_obj.close()

            return float(lines[0].strip('\n')), float(lines[1].strip('\n'))

        def file_to_numpy_matrix(folder, file_name):
            file_obj = open(folder + '/' + file_name + '.txt', 'r')
            lines = np.array(file_obj.readlines())
            file_obj.close()

            def func(line):
                line_ = np.array(line.strip('\n').split(','))
                return line_.astype(float)

            return np.apply_along_axis(func, 1, lines)

        self.air_temperature = file_to_numpy(self.folder, 'temperature')
        self.air_relative_humidity = file_to_numpy(self.folder, 'relative_humidity')
        self.area = file_to_numpy(self.folder, 'area')
        self.height_top, self.height_stratification = get_heights(self.folder)
        self.heat_flux = file_to_numpy_matrix(self.folder, 'heat_flux')
        self.vapour_flux = file_to_numpy_matrix(self.folder, 'vapour_flux')

    def run_row(self, row_index: int):

        # new mean temperature i K
        air_temperature_in_k = celsius_to_kelvin(self.air_temperature[row_index])
        temperature_row = new_mean_temperature(self.area,
                                               self.height_top,
                                               air_temperature_in_k,
                                               self.heat_flux)

        # air flow
        air_flow_row = air_flow(self.area,
                                self.height_top,
                                air_temperature_in_k,
                                temperature_row)

        # new relative humidity
        relative_humidity_row = new_mean_relative_humidity(self.area,
                                                           self.height_top,
                                                           temperature_row,
                                                           relative_humidity_to_vapour_pressure(
                                                               self.air_relative_humidity[row_index],
                                                               air_temperature_in_k),
                                                           self.vapour_flux,
                                                           air_flow_row
                                                           )

        # new stratified relative humidity
        stratified_relative_humidity_row = stratification(self.height_stratification,
                                                          relative_humidity_row,
                                                          self.height_top,
                                                          self.air_relative_humidity[row_index]
                                                          )

        # new stratified temperature in C
        stratified_temperature_row = stratification(self.height_stratification,
                                                    kelvin_to_celsius(temperature_row),
                                                    self.height_top,
                                                    self.air_temperature[row_index])

        # write results
        temp_file = open(self.folder + '/temp_' + str(row_index) + '.txt')
        temp_file.write(','.join(stratified_temperature_row.astype(str)))
        temp_file.close()

        relhum_file = open(self.folder + '/relhum_' + str(row_index) + '.txt')
        relhum_file.write(','.join(stratified_relative_humidity_row.astype(str)))
        relhum_file.close()

        return row_index

    def run_parallel(self):
        self.get_files()

        rows = np.linspace(0,
                           np.size(self.heat_flux, 1),
                           np.size(self.heat_flux, 1) + 1
                           ).astype(int)

        pool = multiprocessing.Pool(processes=self.processes)
        processed_rows = pool.map(self.run_row, rows)

        return processed_rows

    def reconstruct_results(self, processed_rows):
        # Sort row list
        sorted_rows = sorted(processed_rows)

        # open result files
        temperature_file = open(self.folder + '/temperature_results.txt', 'w')
        relhum_file = open(self.folder + '/relative_humidity_results.txt', 'w')

        for row_number in sorted_rows:

            # process temperature files
            temp_path = self.folder + '/temp_' + str(row_number) + '.txt'
            temp_obj = open(temp_path, 'r')
            line = temp_obj.readlines()
            temperature_file.write(line + '\n')
            temp_obj.close()
            os.remove(temp_path)

            # process relative humidity files
            relhum_path = self.folder + '/relhum_' + str(row_number) + '.txt'
            relhum_obj = open(relhum_path, 'r')
            line = relhum_obj.readlines()
            relhum_file.write(line + '\n')
            relhum_obj.close()
            os.remove(relhum_path)

        temperature_file.close()
        relhum_file.close()

        return True

    def run(self):

        #if __name__ == "__main__":
        rows = self.run_parallel()
        self.reconstruct_results(rows)
        return True


def new_mean_relative_humidity(area, height_external, temperature_internal, vapour_pressure_external,
                               vapour_production, air_flow):

    vapour_pressure = new_mean_vapour_pressure(area, height_external, temperature_internal, vapour_pressure_external,
                                               vapour_production, air_flow)

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

    contact = air_flow_ / (gas_constant_vapour * temperature_internal)  # -
    capacity = volume_air / (gas_constant_vapour * temperature_internal)  # -
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

    return area * np.sqrt(2 * abs(delta_pressure) / delta_pressure) * delta_pressure / abs(delta_pressure)


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


def relative_humidity_to_vapour_pressure(relative_humidity, temperature):
    """
    Convert relative humidity to vapour pressure
    :param relative_humidity: unitless
    :param temperature: in K
    :return: vapour pressure in Pa
    """

    temperature_c = kelvin_to_celsius(temperature)  # C
    saturated_pressure = 288.68 * (1.098 + temperature_c/100)**8.02  # Pa
    vapour_pressure = relative_humidity * saturated_pressure  # Pa

    return vapour_pressure


def stratification(height, value_mean, height_top, value_top):
    """
    Calculates the stratification of the temperature or relative humidity
    :param height: height at which the stratification value is wanted. in m.
    :param value_mean: mean value
    :param height_top: height at the top of the boundary. in m
    :param value_top: value at the top of the boundary
    :return: value at desired height.
    """
    return value_mean - 2 * height * (value_mean - value_top)/height_top


def failed_new_temperature(area, height_external, temperature_external, temperature_production):
    """
    Calculates a new temperature and an air exchange
    :param area: in m^2
    :param height_external: in m
    :param temperature_external: in K
    :param temperature_production: in K/s
    :return: temperature in K and air_exchange in m^3/s
    """

    density_air = 1.29 * 273 / temperature_external  # kg/m^3
    specific_heat_capacity = 1005  # J/kgK
    thermal_transmittance = 1  # W/m^2K
    gravity = 9.81  # m/s^2
    volume_air = area * height_external  # m^3
    height_internal = height_external/2  # m

    def air_flow(temperature_internal_):
        """
        Calculates an air flow based on an internal temperature
        :param temperature_internal_: in K
        :return: air flow in m^3/s
        """

        delta_temperature = temperature_external - temperature_internal_
        delta_pressure = density_air * gravity * (height_external - height_internal) * \
                         delta_temperature/temperature_internal_

        return area * np.sqrt(2 * abs(delta_pressure)/delta_pressure) * delta_pressure/abs(delta_pressure)

    def new_temperature_(temperature_internal):
        """
        Solves for a new temperature
        :param temperature_internal: in K
        :return: temperature in K
        """

        air_flow_ = air_flow(temperature_internal)
        contact = thermal_transmittance * area + specific_heat_capacity * air_flow_
        capacity = volume_air * density_air * specific_heat_capacity

        return temperature_external + temperature_production/contact * (1 - np.exp(-contact/capacity)) \
            - temperature_internal

    #temperature = fsolve(new_temperature_, temperature_external)
    #air_flow_ = air_flow(temperature)

    #return temperature, air_flow_