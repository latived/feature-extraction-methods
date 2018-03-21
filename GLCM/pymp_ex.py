import pymp
import numpy as np
import csv

ex_array = pymp.shared.array((100,), dtype='uint8')
with pymp.Parallel(4) as par:
    for index in par.range(0, 100):
        ex_array[index] = index
        par.print(par.thread_num)
"""        
with open('test_par.csv', 'a', newline='') as tfile:
    fieldnames = ['t1']
    writer = csv.DictWriter(tfile, fieldnames=fieldnames)
    writer.writeheader()
    with pymp.Parallel(6) as par:
        for index in par.range(0, 100):
            writer.writerow({'t1':ex_array[index]})
"""
