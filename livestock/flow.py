__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports


# Module imports
import numpy as np
import pysal
import shapefile

# Livestock imports
from livestock import geometry

# ---------------------------------------------------------------------------- #
# Flow Functions and Classes


def flow_from_centers(mesh):
    # get nearest neighbours
    # calculate lowest neighbour
    # create unique paths
    # compute and write paths

    neighbour_matrix = pysal.weights.Queen.from_iterable(mesh)

    for index in range(len(mesh)):
        pass