# Imports
import subprocess

# Functions

def edit_version():
    setup_file = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\setup.py'

    file_obj = open(setup_file, 'r')
    data = file_obj.readlines()
    file_obj.close()

    version = data[16].split('=')[1].strip()[1:-2]
    if version.endswith('a'):
        split = version.split('_')
        alpha = int(split[1][:-1]) + 1
        new_version = split[0] + f"_{alpha}a"
    else:
        new_version = version + "_1a"

    data[16] = "      version='" + new_version + "',\n"

    print(f"Current version: {version}\n"
          f"New version: {new_version}")

    file_obj = open(setup_file, 'w')
    file_obj.writelines(data)
    file_obj.close()


def make_wheel():
    subprocess.call('python setup.py sdist bdist_wheel',
                    cwd=r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock')


edit_version()
make_wheel()