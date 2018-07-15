import numpy as np
from functools import wraps
import os


def file_as_input(function_):

    @wraps(function_)
    def wrapper(*args, **kwargs):

        args_ = []
        for arg in args:
            try:
                os.path.isfile(arg)
                args_.append(np.loadtxt(arg))
            except ValueError:
                args_.append(arg)

        return function_(*args_, **kwargs)
    return wrapper


def poly_list(x):
    return x**2 + 5*x - 2


@file_as_input
def poly_file(x):
    return x**2 + 5*x - 2




data_file = r'C:\Users\Christian\Dropbox\Arbejde\DTU BYG\Livestock\livestock\livestock\test\local_test\data.txt'
data_list = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


print(poly_file(data_list))

print(poly_file(data_file))
