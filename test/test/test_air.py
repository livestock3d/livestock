import air as air
import shutil


def unpack(folder):
    test_files = folder + '/test_files.zip'
    shutil.unpack_archive(test_files, folder)


folder_path = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\livestock\test\test_data\new_air_conditions'

if __name__ == '__main__':
    air.new_temperature_and_relative_humidity(folder_path, 3)
