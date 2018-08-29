import bpy
import bmesh
import json

def clean():
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select = True
        else:
            o.select = False

    # call the operator once
    bpy.ops.object.delete()


clean()

file_obj = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\tests\test_data\drainage_flow\drain_mesh.obj'

bpy.ops.import_scene.obj(filepath=file_obj, axis_forward='X', axis_up='Z', )
imported_mesh = bpy.context.selected_objects[-1]

me = imported_mesh.data
bm = bmesh.new()

# Get a BMesh representation
bm.from_mesh(me)

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

#print(lowest_neighbour)


def get_curve_points(start_index, point_list):

    curve_points = []
    next_index = start_index
    while True:
        pt = point_list[next_index]

        if pt:
            next_index = pt[0]
            curve_points.append(pt[1])
        else:
            return curve_points


curves = []
for face_index in range(len(bm.faces)):
    curves.append(get_curve_points(face_index, lowest_neighbour))

outfile = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\tests\test_data\drainage_flow\result.json'

with open(outfile, 'w') as file:
    json.dump(curves, file)
