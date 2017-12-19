import lib_cmf as lc
import pytest
import shutil


def unpack(folder):
    test_files = folder + '/test_files.zip'
    shutil.unpack_archive(test_files, folder)

@pytest.fixture
def finished_project(folder):
    unpack(folder)
    model = lc.CMFModel(folder)

    return model.run_model()


def test_constant():
    folder_path = '/home/ocni/pycharm/deployed_projects/cmf/test/cmf_weather/constant'
    p = finished_project(folder_path)

    # weather from project
    temp = [t for t in p.meteo_stations[0].T]
    relhum = [rh for rh in p.meteo_stations[0].rHmean]
    wind = [ws for ws in p.meteo_stations[0].Windspeed]
    sun = [ss for ss in p.meteo_stations[0].Sunshine]
    rad = [gr for gr in p.meteo_stations[0].Rs]
    rain = [r for r in p.rainfall_stations[0]]
    ground = [gt for gt in p.meteo_stations[0].Tground]

    # weather from file
    file_temp = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    file_relhum = [0.65000000000000002, 0.65000000000000002, 0.65000000000000002, 0.65000000000000002,
                   0.65000000000000002, 0.65000000000000002, 0.65000000000000002, 0.65000000000000002,
                   0.65000000000000002, 0.65000000000000002, 0.65000000000000002, 0.65000000000000002,
                   0.65000000000000002, 0.65000000000000002, 0.65000000000000002, 0.65000000000000002,
                   0.65000000000000002, 0.65000000000000002, 0.65000000000000002, 0.65000000000000002]
    file_wind = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    file_sun = [0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996,
                0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996,
                0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996,
                0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996, 0.19999999999999996]
    file_rad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    file_rain = [24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0, 24.0,
                 24.0, 24.0, 24.0]
    file_ground = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    assert temp == file_temp
    assert relhum == file_relhum
    assert wind == file_wind
    assert sun == file_sun
    assert rad == file_rad
    assert rain == file_rain
    assert ground == file_ground