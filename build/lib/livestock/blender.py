__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports

# Module imports
try:
    import bpy
except ModuleNotFoundError:
    pass

# Livestock imports


# ---------------------------------------------------------------------------- #
# CMF Functions and Classes


def clean():
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select = True
        else:
            o.select = False

    # call the operator once
    bpy.ops.object.delete()
