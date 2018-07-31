__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports


# Module imports


# Livestock imports
from livestock import flow

# ---------------------------------------------------------------------------- #
# CMF Functions and Classes


def test_flow_from_centers(drain_mesh):

    curves = flow.flow_from_centers(drain_mesh)