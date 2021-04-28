# %% md

## Lens Challenge 2.0 - Classification

# %% md

### Define config

# %%

"""
Deep Bayesian strong lensing code

@author(s): Manuel Blanco Valentín (mbvalentin@cbpf.br)
            Clécio de Bom (clecio@debom.com.br)
            Brian Nord
            Jason Poh
            Luciana Dias
"""

""" Basic Modules """

###### Possible error
###### error OOM - reduction batch size and after limits gpus
import tensorflow as tf
import os, sys
sys.path.append('/home/kayque/LENSLOAD/CH2/')
data_folder = '/share/storage1/arcfinding/challenge2/'

"""
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
"""
import numpy as np

""" Execution time measuring """
from time import time
import time
import matplotlib
from utils import utils
""" keras backend to clear session """
import keras.backend as K
import warnings
warnings.filterwarnings("ignore")
from astropy.io import fits
from keras.utils import Progbar

""" Matplotlib """
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from icecream import ic

import cv2


# %% md

### Define chrono marks

# %%

def add_time_mark(chrono, label):
    tnow = time()
    chrono['marks'][0].append(label)
    chrono['marks'][1].append(tnow)
    chrono['elapsed'][0].append(label)
    if len(chrono['marks'][1]) > 1:
        telapsed = utils.ElapsedTime(chrono['marks'][1][-2])
    else:
        telapsed = utils.ElapsedTime(tnow)
    chrono['elapsed'][1].append(telapsed)


# %% md

### Initialize stuff

# %%

def normalizey(X, vmin=-4e-11, vmax=2.5e-10):
    X = np.clip(X, vmin, vmax)
    X = (X - vmin) / (vmax - vmin)
    X = np.log10(X.astype(np.float16) + 1e-10)
    mmin = X.min()
    mmax = X.max()
    X = (X - mmin) / (mmax - mmin)
    return X

def DataGeneratorCh2(num_samples, version, input_shape, input_shape_vis, using_238):
    print(" ** Using Manu's generator")
    foldtimer = time.perf_counter()

    # %% md

    ### Load data

    # %%

    data_dir = '/home/kayque/LENSLOAD/CH2/'
    catalog_name = 'image_catalog2.0train.csv'

    """ Load catalog before images """
    import pandas as pd

    data_folder = '/share/storage1/arcfinding/challenge2/'

    if using_238:
        data_folder = '/home/dados2T/DataChallenge2/'
        tr = 'Train'
        head = 0
        catalog_name = 'image_catalog2.0train_corrigido2.csv'
    else:
        tr = 'train'
        head = 28
    catalog = pd.read_csv(os.path.join(data_dir, (data_folder+catalog_name)), header=head)  # 28 for old catalog

    """ Now load images using catalog's IDs """
    from skimage.transform import resize

    channels = ['H', 'J', 'Y']
    #channels = ['VIS', 'J', 'Y']
    channel_vis = ['VIS']
    #nsamples = len(catalog['ID'])
    print(channels, channel_vis)
    idxs2keep = []
    #missing_data = [13913, 26305, 33597, 44071, 59871, 61145, 70458, 88731, 94173]
    missing_data = [13912, 26304, 33596, 44070, 59870, 61144, 70457, 88730, 94172]
    for a in missing_data:
        labels = catalog['ID'].drop(a)
    labels = labels[0:num_samples]
    nsamples = len(labels)

    images = np.zeros((len(labels), input_shape, input_shape, 3))
    images_vis = np.zeros((len(labels), input_shape_vis, input_shape_vis, 3))
    """ Loop thru indexes """
    with tf.device('/GPU:0'):
        pbar = Progbar(nsamples - 1)
        for iid, cid in enumerate(labels):  # enumerate(labels):

            """ Loop thru channels"""
            for ich, ch in enumerate(channels):

                """ Init image dir and name """
                image_file = os.path.join(data_folder,
                                      tr,
                                      'Public',
                                      'EUC_' + ch,
                                      'imageEUC_{}-{}.fits'.format(ch, cid))

                if os.path.isfile(image_file):
                        #print(image_galsub_file)

                        #if os.path.isfile(image_file):# and os.path.isfile(image_galsub_file):

                    """ Import data with astropy """
                    image_data = fits.getdata(image_file, ext=0)
                    image_data = resize(image_data, (input_shape,input_shape))

                    """ Initialize images array in case we haven't done it yet """
                    if images is None:
                        images = np.zeros((nsamples, *image_data.shape, len(channels)))

                    """ Set data in array """
                    image_data[np.where(np.isnan(image_data))] = 0
                    image_data = utils.center_crop_and_resize(image_data, input_shape)
                    images[iid,:,:,ich] = image_data
                    if iid not in idxs2keep:
                        idxs2keep.append(iid)
                else:
                    print('\tSkipping index: {} (ID: {})'.format(iid, cid))
                    break

            for ich, ch in enumerate(channel_vis):
                """ Init image dir and name """
                image_file_vis = os.path.join(data_folder,
                                       tr,
                                      'Public',
                                      'EUC_' + ch,
                                      'imageEUC_{}-{}.fits'.format(ch, cid))

                if os.path.isfile(image_file_vis):
                    #print(image_galsub_file)

                    #if os.path.isfile(image_file):# and os.path.isfile(image_galsub_file):

                    """ Import data with astropy """
                    image_data_vis = fits.getdata(image_file_vis, ext=0)
                    image_data_vis = resize(image_data_vis, (input_shape_vis,input_shape_vis))

                    """ Initialize images array in case we haven't done it yet """
                    if images_vis is None:
                        images_vis = np.zeros((nsamples, *image_data_vis.shape, len(channels)))

                    """ Set data in array """
                    image_data_vis[np.where(np.isnan(image_data_vis))] = 0
                    image_data_vis = utils.center_crop_and_resize(image_data_vis, input_shape_vis)
                    images_vis[iid,:,:,0] = image_data_vis
                    images_vis[iid,:,:,1] = image_data_vis
                    images_vis[iid,:,:,2] = image_data_vis
                    if iid not in idxs2keep:
                        idxs2keep.append(iid)
                else:
                    print('\tSkipping index: {} (ID: {})'.format(iid, cid))
                    break

            if iid % 1000 == 0 and iid != 0:
                pbar.update(iid)

        apply_log = True
        print('\n -- Normalizing...')
        images = utils.normalize2(images, len(channels), channels, apply_log=apply_log, vis=False)
        print('\n -- Normalizing VIS...')
        images_vis = utils.normalize2(images_vis, len(channels), channel_vis, apply_log=apply_log, vis=True)
        print(apply_log)
    np.random.shuffle(idxs2keep)
    #images = images.astype(np.float16)
    catalog = catalog.loc[idxs2keep]
    images = images[idxs2keep]
    images_vis = images_vis[idxs2keep]
    #images = images[:num_samples]
    catalog = catalog[:num_samples]
    print(len(images), len(images_vis), len(catalog))

    fig = plt.figure(figsize=(10,3))
    NN = images.shape[-1]
    for i in range(NN):
        plt.subplot(1,NN,i+1)
    _ = plt.hist(np.clip(images[:,:,:,i].flatten(),-0.4e-10,2.5e-10),bins=256)
    _ = plt.title(channels[i])
    fig.savefig('histogram_for_normalization.png')

    is_lens = (catalog['n_source_im'] > 0) & (catalog['mag_eff'] > 1.6) & (catalog['n_pix_source'] > 20)  # 700
    is_lens = 1.0 * is_lens
    #is_lens = to_categorprintal(is_lens, 2)
    print(catalog['ID'])
    print(is_lens)

    inputs = images[0:int(0.9*len(is_lens)),:,:,:]#.astype(np.float16) 
    inputs_vis = images_vis[0:int(0.9*len(is_lens)),:,:,:]
    outputs = is_lens[0:int(0.9*len(is_lens))].to_numpy()
    test_inps = images[int(0.9*len(is_lens)):int(len(is_lens)), :, :, :]
    test_outs = is_lens[int(0.9*len(is_lens)):int(len(is_lens))]
    test_inps_v = images_vis[int(0.9*len(is_lens)):int(len(is_lens)), :, :, :]
    
    #images = utils.normalize(images)
    print(inputs[0,:,:,:])
    print(inputs_vis[0, :, :, :])
    #print(outputs)

    index = utils.save_clue(inputs, outputs, num_samples, version, 'hjy', input_shape, 5, 5, 0, channels)
    index = utils.save_clue(inputs_vis, outputs, num_samples, version, 'vis', input_shape_vis, 5, 5, 0, channels)
    #inputs = {'images': inputs}
    #outputs = {'is_lens': outputs}
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Data Generation TIME: %.3f minutes.' % elaps)
    return (inputs, inputs_vis, outputs, index, channels, test_inps, test_inps_v, test_outs)

def DataGeneratorCh2_ver_2(num_samples, version, input_shape, input_shape_vis):
    from sklearn.model_selection import train_test_split
    data_dir = '/home/dados2T/DataChallenge2/'
    sys.path.append(data_dir)

    images_hjy = np.load(os.path.join(data_dir,'images_hjy_normalized.npy'))
    images_vis = np.load(os.path.join(data_dir,'images_vis_normalized.npy'))
    pad = np.zeros((images_vis.shape[0],images_vis.shape[1],images_vis.shape[2],1), dtype="float32")
    is_lens = np.load(os.path.join(data_dir,'Y.npy'))

    X_train_hjy, X_test_hjy, Y_train, Y_test = train_test_split(images_hjy, is_lens, test_size = 0.10, random_state = 7)
    X_train_vis, X_test_vis, Y_train, Y_test = train_test_split(np.concatenate([images_vis[:,:,:,2:],pad,pad], axis=-1), is_lens, test_size = 0.10, random_state = 7)

    ic(X_train_vis, X_test_vis)

    Y_train = Y_train[:, 1]
    Y_test = Y_test[:, 1]
    ic(Y_train, Y_test)
    channels = ['H', 'J', 'Y']

    ic(X_train_hjy.shape, X_test_hjy.shape)
    ic(X_train_vis.shape, X_test_vis.shape)

    index = utils.save_clue(X_train_hjy, Y_train, num_samples, version, 'hjy', input_shape, 5, 5, 0, channels)
    #index = utils.save_clue(X_train_vis, Y_train, num_samples, version, 'vis', input_shape_vis, 5, 5, 0, channels)

    return (X_train_hjy, X_train_vis, Y_train, index, channels, X_test_hjy, X_test_vis, Y_test)