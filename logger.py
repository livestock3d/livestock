__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import logging


# Livestock imports

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Loggers


def logger():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream.setFormatter(formatter)

    # FileHandlers
    file_info = logging.FileHandler(f'{__name__}_info.log')
    file_info.setLevel(logging.INFO)
    file_info.setFormatter(formatter)

    file_debug = logging.FileHandler(f'{__name__}_debug.log')
    file_debug.setLevel(logging.DEBUG)
    file_debug.setFormatter(formatter)

    log.addHandler(stream)
    log.addHandler(file_debug)
    log.addHandler(file_info)

    return log