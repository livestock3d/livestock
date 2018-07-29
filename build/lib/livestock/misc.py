__author__ = "Christian Kongsgaard"
__license__ = "MIT"
__version__ = "0.0.1"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import os
import shutil
import subprocess

# Livestock imports


# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Miscellaneous Library


def run_cfd(files_path):
    """Runs a OpenFoam case"""

    # Get files
    zip_file = files_path + '/cfd_case.zip'
    file_obj = open(files_path + '/cfd_commands.txt', 'r')
    lines = file_obj.readlines()
    file_obj.close()
    os.remove(files_path + '/cfd_commands.txt')

    commands = lines[0][:-1].split(',')
    cpus = lines[1].split(',')

    # unpack and delete zip
    shutil.unpack_archive(zip_file, files_path)
    os.remove(zip_file)

    # run openFoam commands
    for i in range(0, len(commands)):
        bash_command = str(commands[i])
        thread = subprocess.Popen(bash_command)
        thread.wait()
        thread.kill()

    # zip result
    shutil.make_archive(files_path + '/solved_cfd_case', 'zip', files_path)
