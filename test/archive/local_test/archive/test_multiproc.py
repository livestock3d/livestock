import multiprocessing
import numpy as np

def doubler(number):
    return number[0] * 2


numbers = [[5,0], [10,0], [20,]]


if __name__ == '__main__':

    pool = multiprocessing.Pool(processes=3)
    result = pool.map(doubler, numbers)
    print(result)