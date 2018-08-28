__author__ = "Christian Kongsgaard"
__license__ = "GNU GPLv3"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import logging
import os
import shutil
import platform

# Livestock imports

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Loggers


def log_path():
    if not platform.system() == 'Linux':
        livestock_path = r'C:\livestock'
    else:
        livestock_path = os.path.dirname(__file__)

    log_folder = os.path.join(livestock_path, 'logs')

    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    return log_folder


def livestock_logger():

    if os.listdir(log_path()):
        shutil.rmtree(log_path())

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s'
                                  ' - %(message)s')

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    stream.setFormatter(stream_formatter)

    # FileHandlers
    file_info = logging.FileHandler(os.path.join(log_path(),
                                                 'livestock_info.log'))
    file_info.setLevel(logging.INFO)
    file_info.setFormatter(formatter)

    file_debug = logging.FileHandler(os.path.join(log_path(),
                                                  'livestock_debug.log'))
    file_debug.setLevel(logging.DEBUG)
    file_debug.setFormatter(formatter)

    log.addHandler(stream)
    log.addHandler(file_debug)
    log.addHandler(file_info)

    return log
