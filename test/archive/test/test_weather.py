import lib_cmf as lc
from matplotlib import pyplot as plt
import numpy as np
import shutil


def unpack(folder):
    test_files = folder + '/test_files.zip'
    shutil.unpack_archive(test_files, folder)


def get_ylabel(weather_type):
    if str(weather_type) == 'temp':
        return 'Temperature in C'
    elif weather_type == 'relhum':
        return 'Relative Humidity in -'
    elif weather_type == 'wind':
        return 'wind speed in m/s'
    elif weather_type == 'sun':
        return 'Sunshine fraction in -'
    elif weather_type == 'rad':
        return 'Radiation in MJ/m2day'
    elif weather_type == 'rain':
        return 'Rain in mm/day'
    elif weather_type == 'ground':
        return 'Ground temperature in C'


def plot_weather(weather, weather_type, test_type):
    x = np.linspace(0, len(weather), len(weather))

    unit = get_ylabel(weather_type)
    plt.figure(weather_type)
    plt.plot(x, weather)
    plt.title(test_type + ' Weather')
    plt.xlabel('Time in h')
    plt.ylabel(unit)
    plt.show()


def plot_weather_double(weather_double, weather_type, test_type):
    x = np.linspace(0, len(weather_double[0]), len(weather_double[0]))

    unit = get_ylabel(weather_type)
    plt.figure(weather_type)
    plt.plot(x, weather_double[0])
    plt.plot(x, weather_double[1])
    plt.title(test_type + ' Weather')
    plt.xlabel('Time in h')
    plt.ylabel(unit)
    plt.show()


def test_constant():
    folder_path = '/home/ocni/pycharm/deployed_projects/livestock_cmf/test/cmf_weather/constant'

    unpack(folder_path)
    model = lc.CMFModel(folder_path)
    p = model.run_model()
    temp = [t for t in p.meteo_stations[0].T]
    relhum = [rh for rh in p.meteo_stations[0].rHmean]
    wind = [ws for ws in p.meteo_stations[0].Windspeed]
    sun = [ss for ss in p.meteo_stations[0].Sunshine]
    rad = [gr for gr in p.meteo_stations[0].Rs]
    rain = [r for r in p.rainfall_stations[0].data]
    ground = [gt for gt in p.meteo_stations[0].Tground]

    plot_weather(temp, 'temp', 'Constant')
    plot_weather(relhum, 'relhum', 'Constant')
    plot_weather(wind, 'wind', 'Constant')
    plot_weather(sun, 'sun', 'Constant')
    plot_weather(rad, 'rad', 'Constant')
    plot_weather(rain, 'rain', 'Constant')
    plot_weather(ground, 'ground', 'Constant')


def test_sinus():
    folder_path = '/home/ocni/pycharm/deployed_projects/livestock_cmf/test/cmf_weather/sinus'

    unpack(folder_path)
    model = lc.CMFModel(folder_path)
    p = model.run_model()
    temp = [t for t in p.meteo_stations[0].T]
    relhum = [rh for rh in p.meteo_stations[0].rHmean]
    wind = [ws for ws in p.meteo_stations[0].Windspeed]
    sun = [ss for ss in p.meteo_stations[0].Sunshine]
    rad = [gr for gr in p.meteo_stations[0].Rs]
    rain = [r for r in p.rainfall_stations[0].data]
    ground = [gt for gt in p.meteo_stations[0].Tground]

    plot_weather(temp, 'temp', 'Sinus')
    plot_weather(relhum, 'relhum', 'Sinus')
    plot_weather(wind, 'wind', 'Sinus')
    plot_weather(sun, 'sun', 'Sinus')
    plot_weather(rad, 'rad', 'Sinus')
    plot_weather(rain, 'rain', 'Sinus')
    plot_weather(ground, 'ground', 'Sinus')


def test_double():
    folder_path = '/home/ocni/pycharm/deployed_projects/livestock_cmf/test/cmf_weather/double'

    unpack(folder_path)
    model = lc.CMFModel(folder_path)
    p = model.run_model()
    temp = [[t for t in p.meteo_stations[0].T],[t for t in p.meteo_stations[1].T]]
    relhum = [[rh for rh in p.meteo_stations[0].rHmean],[rh for rh in p.meteo_stations[1].rHmean]]
    wind = [[ws for ws in p.meteo_stations[0].Windspeed],[ws for ws in p.meteo_stations[1].Windspeed]]
    sun = [[ss for ss in p.meteo_stations[0].Sunshine],[ss for ss in p.meteo_stations[1].Sunshine]]
    rad = [[gr for gr in p.meteo_stations[0].Rs],[gr for gr in p.meteo_stations[1].Rs]]
    rain = [[r for r in (p.rainfall_stations[0].data)],[r for r in (p.rainfall_stations[1].data)]]
    ground = [[gt for gt in p.meteo_stations[0].Tground],[gt for gt in p.meteo_stations[1].Tground]]

    plot_weather_double(temp, 'temp', 'Double')
    plot_weather_double(relhum, 'relhum', 'Double')
    plot_weather_double(wind, 'wind', 'Double')
    plot_weather_double(sun, 'sun', 'Double')
    plot_weather_double(rad, 'rad', 'Double')
    plot_weather_double(rain, 'rain', 'Double')
    plot_weather_double(ground, 'ground', 'Double')


#test_constant()
#test_sinus()
test_double()
