import shutil


def unpack(folder):
    test_files = folder + '/test_files.zip'
    shutil.unpack_archive(test_files, folder)