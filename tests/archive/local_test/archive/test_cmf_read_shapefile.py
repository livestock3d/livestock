import cmf
from cmf.geos_shapereader import Shapefile as cmf_shape

p=cmf.project()
r_curve=cmf.VanGenuchtenMualem()

# Load cell polygons from file
path = r'C:\Users\Christian\Desktop\Livestock\shapefiles\shape_mesh.shp'
polygons=cmf_shape(path)

# create cell mesh from polygons
cells = p.cells_from_polygons(polygons)

for c in p.cells:
    # Add ten layers
    for i in range(10):
        c.AddLayer((i+1)*0.1,r_curve)
    cmf.Richards.use_for_cell(c)
    c.surfacewater_as_storage()
# subsurface flow
cmf.connect_cells_with_flux(p,cmf.Richards_lateral)
# surface runoff
cmf.connect_cells_with_flux(p,cmf.Manning_Kinematic)