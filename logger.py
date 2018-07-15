__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import logging


# Livestock imports

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Loggers

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# StreamHandler
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
stream.setFormatter(formatter)

# FileHandlers
file_info = logging.FileHandler(f'{name}_info.log')
file_info.setLevel(logging.INFO)
file_info.setFormatter(formatter)

file_debug = logging.FileHandler(f'{name}_debug.log')
file_debug.setLevel(logging.DEBUG)
file_debug.setFormatter(formatter)

logger.addHandler(stream)
logger.addHandler(file_debug)
logger.addHandler(file_info)