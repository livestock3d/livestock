# Imports
from pathlib import Path
home_user = str(Path.home())
from livestock.hydrology import CMFModel
# Run CMF Model
folder = home_user + '/livestock/ssh'
model = CMFModel(folder)
model.run_model()
# Announce that template finished and create out file
print('Finished with template')
file_obj = open('out.txt', 'w')
file_obj.close()