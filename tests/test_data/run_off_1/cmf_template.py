# Imports
import sys
sys.path.insert(0, r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock')
from livestock.hydrology import run_model
# Run CMF Model
run_model(r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock_gh\grasshopper\tests\test_data\CMF_Slope_sec_v0\cmf')
# Announce that template finished and create out file
print('Finished with template')
file_obj = open('out.txt', 'w')
file_obj.close()