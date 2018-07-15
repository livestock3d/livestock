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

imported_mesh = bpy.ops.import_scene.obj(filepath=file_obj, axis_forward='X', axis_up='Z',)
mesh = bmesh.new()
mesh.from_mesh(imported_mesh)

for face in mesh.faces:
    print(face.clac_center_median())