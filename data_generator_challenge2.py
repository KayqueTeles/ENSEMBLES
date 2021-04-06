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

def DataGenerator(num_samples, version, input_shape):
    ###STEP = TRAIN OR TEST; 
    print(" ** Initiating My DataGenerator")
    foldtimer = time.perf_counter()
    Path('/share/storage1/arcfinding/challenge2/').parent
    os.chdir('/share/storage1/arcfinding/challenge2/')
    var = os.getcwd()
    print(" ** Searching for data at: ", var)

    train_folder = var + '/train/Public/'
    labels_csv = var + '/image_catalog2.0train.csv'

    y_open = pd.read_csv(labels_csv, delimiter=',', skiprows = 28, header=0, low_memory=False)
    n_pix_source_pos = np.array(y_open.loc[y_open['n_pix_source'] >20])   ###SOMEHOW, IT READS VALUES DIFFERENTLY - THE DATASET IS FIXED
    print(' ** Len_pix_source number: ', len(n_pix_source_pos))
    n_pix_source_pos = n_pix_source_pos[:,1]
    print(n_pix_source_pos)
    n_source_im_pos = np.array(y_open.loc[y_open['n_source_im'] >0])
    print(' ** Len_source_img number: ', len(n_source_im_pos))
    n_source_im_pos = n_source_im_pos[:,1]
    print(n_source_im_pos)
    mag_eff_pos = np.array(y_open.loc[y_open['mag_eff'] >1.6])
    print(' ** Len_pix_source number: ', len(mag_eff_pos))
    mag_eff_pos = mag_eff_pos[:,1]
    print(mag_eff_pos)
    positives = []
    print(" ** Gathering positive cases from data...")
    inter = utils.intersect1d(n_pix_source_pos, n_source_im_pos)
    positives = utils.intersect1d(inter, mag_eff_pos)
    print(" ** Number of lensing Positives:", len(positives))
    print(positives)

    y_data = []
    y_data = np.array(y_data)
    PARAM = 200001
    y = 0
    for x in range(num_samples):
        #print(y_data)
        if positives[y] == (PARAM+x):
            y_data = np.append(y_data, [1])
            y = y + 1
        else:
            y_data = np.append(y_data, [0])
    y_data = np.asarray(y_data)
    print(y_data)
    print(" ** Y_data size isss....", len(y_data))

    #train_folders = ['EUC_H', 'EUC_J', 'EUC_VIS', 'EUC_Y']
    train_folders = ['EUC_H', 'EUC_VIS', 'EUC_Y']
    channels = ['H', 'VIS', 'Y']
    #train_folders = ['EUC_H', 'EUC_J', 'EUC_Y']
    print(" ** Working with bands: ", train_folders)

    resizeds = 0
    c = 0
    img_shp = [input_shape, input_shape]
    num_channels = len(train_folders)
    x_data = np.zeros((num_samples, img_shp[0],img_shp[1],num_channels))
    missing_data = [13913, 26305, 33597, 44071, 59871, 61145, 70458, 88731, 94173]

    for band in train_folders:
        train_file_list = os.listdir(train_folder + '/' + band)
        #print(train_file_list)
        #print(band)
        index = 0
        for num, fit in zip(range(num_samples), train_file_list):
            if num in missing_data:
                print(' ** Detected missing_data.')
            #if os.path.isfile(train_folder + band + '/' + 'image' + band + '-%s.fits' % (PARAM+fit)):
            else:
                #print(fit)
                x_open = fits.getdata(train_folder + band + '/' + fit, ext=0)
                #x_open = fits.getdata(train_folder + band + '/' + 'image' + band + '-%s.fits' % (PARAM+fit), ext=0)
                x_file = np.array(x_open)
                if x_file.shape[0] != input_shape:
                    x_file = np.resize(x_file,(input_shape,input_shape))
                    resizeds = resizeds + 1
                x_data = utils.normalize(x_data.astype(np.float16))
                x_data[np.where(np.isnan(x_data))] = 0
                #print(x_file)
                #print('***************************')
                x_data[index,:,:,c] = x_file
                index = index+1
        c = c+1

    print(x_data.shape)
    print(" ** Resized cases: ", resizeds)
    print(x_data[0,:,:,1])
    print(x_data[0,:,:,2])
    print(x_data[0,:,:,0])
    Path('/home/kayque/LENSLOAD/').parent
    os.chdir('/home/kayque/LENSLOAD/')
    index = utils.save_clue(x_data, y_data, num_samples, version, 'generator', img_shp[0], 5, 5, 0)
    #mmin = x_data.min()
    #mmax = x_data.max()
    #x_data = utils.normalize(x_data.astype(np.float16))
    #x_data[np.where(np.isnan(x_data))] = 0
    #x_data = (x_data - mmin)/(mmax - mmin)
    #x_data = (x_data - mmin)/(mmax - mmin)
    #x_data = cv2.normalize(x_data, None, 0, 1, cv2.NORM_MINMAX)
    #x_data = utils.normalize(x_data)
    #x_data = utils.normalize(x_data)
    index = utils.save_clue(x_data, y_data, num_samples, version, 'generator', img_shp[0], 5, 5, index, channels)
    print(" ** Initiating normalization check.")
    print(x_data.shape)
    print(x_data[0,:,:,1])
    print(x_data[0,:,:,2])
    print(x_data[0,:,:,0])
    #print_images_clecio_like(x_data=x_data, num_prints=10, num_channels=num_channels, input_shape=66)
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Data Generation TIME: %.3f minutes.' % elaps)

    return (x_data, y_data, index, channels)

    #x_data = (x_data - np.min(x_data))/(np.max(x_data)+np.min(x_data))
    #x_data = cv2.normalize(x_data, None, 0, 255, cv2.NORM_MINMAX)
    #x_data = cv2.normalize(x_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #x_file = utils.imadjust(np.uint8(x_file),vin=[np.min(x_file),np.max(x_file)], vout=(0, 255))
    #x_data = (x_data - np.nanmin(x_data))/np.ptp(x_data)