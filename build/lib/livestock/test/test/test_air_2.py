from air import *
import numpy as np


temp = np.array([30, 25])
temp_k = celsius_to_kelvin(temp)
rh = np.array([0.8, 0.6])
vapour = np.array([0.1, 0.1])
volume = np.array([10, 10])

result = compute_temperature_relative_humidity(temp_k, rh, vapour, volume)

print('Current Temp', temp_k)
print('New Temp', result[0])
print('Current RH', rh)
print('New RH', result[1])
print('Available Flux', vapour)
print('Used flux', result[3])