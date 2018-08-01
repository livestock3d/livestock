__author__ = "Christian Kongsgaard"
__license__ = "MIT"

# ---------------------------------------------------------------------------- #
# Imports


# Module imports
try:
    import bpy
    import bmesh
    import json
    import os
    import typing
except ModuleNotFoundError:
    pass

# Livestock imports
from livestock import blender

# ---------------------------------------------------------------------------- #
# Flow Functions and Classes


def get_curve_points(start_point, start_index: int, point_list) \
        -> typing.List[tuple]:

    curve_points = [start_point, ]
    next_index = start_index

    while True:
        pt = point_list[next_index]

        if pt:
            next_index = pt[0]
            curve_points.append(pt[1])
        else:
            return curve_points


def flow_from_centers(folder: str) -> None:
    blender.clean()

    mesh_file = os.path.join(folder, 'drain_mesh.obj')
    result_file = os.path.join(folder, 'results.json')
    bpy.ops.import_scene.obj(filepath=mesh_file, axis_forward='X', axis_up='Z')
    imported_mesh = bpy.context.selected_objects[-1]

    # Get a BMesh representation
    me = imported_mesh.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()

    lowest_neighbour = []
    for face in bm.faces:
        linked_faces = set(f
                           for v in face.verts
                           for f in v.link_faces)
        centers = [[linked_face.index, tuple(linked_face.calc_center_median())]
                   for linked_face in linked_faces]

        sorted_centers = sorted(centers, key=lambda v: v[1][2])

        if face.calc_center_median().z <= sorted_centers[0][1][2]:
            lowest_neighbour.append(None)
        else:
            lowest_neighbour.append(sorted_centers[0])

    curves = []
    for face_index in range(len(bm.faces)):
        start_point = tuple(bm.faces[face_index].calc_center_median())
        curves.append(get_curve_points(start_point, face_index,
                                       lowest_neighbour))

    with open(result_file, 'w') as outfile:
        json.dump(curves, outfile)
