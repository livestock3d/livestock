# Imports
from livestock.hydrology import run_model
# Run CMF Model
run_model(r'C:\Users\ocni\Dropbox\Arbejde\DTU BYG\Livestock\livestock_gh\grasshopper\tests\test_data\Weather_v2\cmf')
# Announce that template finished and create out file
print('Finished with template')
file_obj = open('out.txt', 'w')
file_obj.close()