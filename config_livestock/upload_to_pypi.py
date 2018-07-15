# Imports
import os
import subprocess
import datetime

# Functions
def delete_old_files():
    dist_folder = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\dist'

    dist_files = os.listdir(dist_folder)

    for f in dist_files:
        os.remove(dist_folder + '/' + f)


def edit_version():
    setup_file = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\setup.py'

    file_obj = open(setup_file, 'r')
    data = file_obj.readlines()
    file_obj.close()

    version = data[16].split('=')[1].strip()[1:-2]
    date_version = datetime.datetime.now().strftime('%Y.%m')
    old_version_parts = version.split('.')
    if date_version == old_version_parts[0] + '.' + old_version_parts[1]:
        subversion = int(old_version_parts[2].split('_')[0]) + 1
        new_version = date_version + '.' + str(subversion)
    else:
        new_version = date_version + '.01'

    data[16] = "      version='" + new_version + "',\n"

    print(f"Current version: {version}\n"
          f"New version: {new_version}")

    file_obj = open(setup_file, 'w')
    file_obj.writelines(data)
    file_obj.close()


def call_upload():
    subprocess.call('python setup.py sdist bdist_wheel',
                    cwd=r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock')

    subprocess.call('twine upload dist/*',
                    cwd=r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock')


# Run
delete_old_files()
edit_version()
call_upload()
