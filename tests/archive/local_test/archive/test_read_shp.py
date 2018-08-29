import geopandas

path = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\livestock\test\test_data\obj_to_shp\shape_mesh.shp'

mesh = geopandas.read_file(path)

print(mesh)