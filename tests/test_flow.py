__author__ = "Christian Kongsgaard"
__license__ = "GNU GPLv3"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports


# Module imports
import pytest
import platform
import os

# Livestock imports
from livestock import flow

# -------------------------------------------------------------------------------------------------------------------- #
# CMF Functions and Classes


@pytest.mark.skipif(platform.system() == 'Linux',
                    reason='Test should only run locally')
def test_flow_from_centers(drain_mesh):

    flow.flow_from_centers(drain_mesh)

    assert os.path.exists(os.path.join(drain_mesh, 'results.json'))
