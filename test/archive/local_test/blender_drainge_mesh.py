import bpy
import bmesh

def clean():
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.select = True
        else:
            o.select = False

    # call the operator once
    bpy.ops.object.delete()
        
clean()
file_obj = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\livestock\test\test_data\drainage_flow\drain_mesh.obj'

bpy.ops.import_scene.obj(filepath=file_obj, axis_forward='X', axis_up='Z',)
imported_mesh = bpy.context.selected_objects[-1]

me = imported_mesh.data
bm = bmesh.new()

# Get a BMesh representation
bm.from_mesh(me)

for face in bm.faces:
    print(face.calc_center_median())