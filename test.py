import numpy as np


a = np.array([[
    [.5, .5, .5],
    [.5, .5, .5],
    [.5, .5, .5]],

    [[1, .5, .5],
    [.6, .6, .6],
    [.7, .7, .7],
]])

print(np.var(a, axis=0)*4)



