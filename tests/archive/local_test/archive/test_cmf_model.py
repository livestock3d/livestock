import hydrology as hy
import cmf
import shutil

def unpack(folder):
    test_files = folder + '/test_files.zip'
    shutil.unpack_archive(test_files, folder)

folder_path = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\livestock\test\test_data\cmf_slope'
unpack(folder_path)

model = hy.CMFModel(folder_path)
case = model.run_model()

cmf.describe(case, out=open(folder_path + '/case_description.txt','w'))
