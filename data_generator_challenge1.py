import numpy as np
import os
import pandas as pd
from pathlib import Path
from astropy.io import fits
import time
import cv2
import csv
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import os
import pydot
import pydot_ng
import matplotlib.pyplot as plt
from astropy.io import fits
from pathlib import Path
from utils import utils

def DataGenerator(num_samples, version):
    ###STEP = TRAIN OR TEST; 
    print(" ** Initiating DataGenerator")
    Path('/home/kayque/LENSLOAD/').parent
    os.chdir('/home/kayque/LENSLOAD/')
    foldtimer = time.perf_counter()
    print('\n ** Starting data preprocessing...')
    labels = pd.read_csv(var + 'y_data20000fits.csv', delimiter=',', header=None)
    y_data = np.array(labels, np.uint8)
    y_size = len(y_data)
    print(y_data)

    x_datasaved = h5py.File(var + 'x_data20000fits.h5', 'r')
    Ni_channels = 0  # first channel
    N_channels = 3  # number of channels

    x_data = x_datasaved['data']
    x_size = len(x_data)
    x_data = x_data[:, :, :, Ni_channels:Ni_channels + N_channels]
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Data Generation TIME: %.3f minutes.' % elaps)

    return (x_data, y_data, index)
