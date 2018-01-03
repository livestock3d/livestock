import hydrology as hy
import cmf
import test.helper_functions as helper


folder_path = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\livestock\test\test_data\cmf_slope'
helper.unpack(folder_path)

model = hy.CMFModel(folder_path)
case = model.run_model()

cmf.describe(case, out=open(folder_path + '/case_description.txt', 'w'))
