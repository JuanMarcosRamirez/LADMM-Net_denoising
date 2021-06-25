# Generate noisy image data set
# Juan Marcos Ramirez Rondon, Universidad Rey Juan Carlos, Spain

import numpy as np

def normalize_data(Ii):
    Ii.astype(float)
    a       = np.amin(Ii)
    b       = np.amax(Ii)
    scale   = 1.0 / (b - a)
    Io = np.multiply(scale, np.subtract(Ii,a))
    return Io