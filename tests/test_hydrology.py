__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# -------------------------------------------------------------------------------------------------------------------- #
# Imports

# Module imports
import cmf

# Livestock imports
from livestock import hydrology

# -------------------------------------------------------------------------------------------------------------------- #
# Livestock Test


def test_mesh_to_cells(obj_file_paths):
    project = cmf.project()

    hydrology.mesh_to_cells(project, obj_file_paths, False)

    assert project
    assert project.cells



