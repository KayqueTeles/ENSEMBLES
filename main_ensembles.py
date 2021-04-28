import warnings
import tensorflow as tf
from pathlib import Path
import os
import numpy as np
import shutil
import csv
import cv2
import time
import h5py
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from keras import backend as k_back
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from tensorflow.keras import backend as K
import numpy as np
from IPython.display import Image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_auc_score
from PIL import Image
import bisect
from opt import RAdam
from keras.applications.resnet50 import ResNet50
# Todo - Testar a resnet 50 V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from sklearn.model_selection import train_test_split
#tf.debugging.set_log_device_placement(True)

from utils import files_changer, utils, graphs
import model_lib
from icecream import ic

warnings.filterwarnings('ignore')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

mirrored_strategy = tf.distribute.MirroredStrategy()

ic(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
ic(gpus)
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    ic(len(gpus), len(logical_gpus))
  except:
    # Visible devices must be set before GPUs have been initialized
    print(" -- Couldn't find GPUs")

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

###############################################################################################
#DEFINE CODE'S PARAMETERS
###############################################################################################

version = 62 # EXECUTION VERSION
num_classes = 2  # NUMBER OF OUTPUT CLASSES
rede = 'ensemble'     ##OPTIONS: 'resnet', 'ensemble' or 'effnet'
weights = 'imagenet'   #'imagenet' or None
using_238 = False
preload_weights = False
use_alternative_model = False
use_new_model = True
ch2_testing = False
ch1_testing = False
ch1_weights = False
aug_data = True     ##DO YOU WANT SOME DATA AUGMENTATION?
aug_type = ['rotation_range=180, horizontal_flip=True, vertical_flip=True, zoom_range=0.20, fill_mode="nearest"']
learning_rate = 0.001
optimizer = 'RAdam'   #(clipnorm=0.01)  ##CHOOSE 'sgd' or 'adam' or 'nadam' RAdam()
avgpool = True
dropout = False
loss = 'categorical_crossentropy'  # 'categorical_crossentropy' #'fbeta'  #'binary_crossentropy' usually
loss_regularization = False   ###THIS APPARENTLY INCREASES THE VALIDATION LOSS
num_epochs = 500  # NUMBER OF EPOCHS
batch_size = 64  # Tamanho do Batch
k_folds = 10  # NÚMERO DE FOLDS
percent_data = 1.0  # Porcentagem de Dados DE TESTE a ser usado - TODO - Definir como será utilizado
vallim = 2000  # Quantidade de dados de validação
challenge = 'challenge2'
##about types of ensembles used
dirichlet = False
logistic = False
integrated_stacked = False
dataset_size = 20000
input_shape = 66
input_shape_vis = 200
classes = ['lens', 'not-lens']
#model_list = ['effnet_B2', 'effnet_B4', 'effnet_B6', 'effnet_B7']
model_list = ['effnet_B2']
#model_list = ['resnet50', 'resnet101', 'resnet152', 'effnet_B0', 'effnet_B1', 'effnet_B2', 'effnet_B3', 'effnet_B4', 'effnet_B5', 'effnet_B6', 'effnet_B7', 'inceptionV2', 'inceptionV3', 'xception']

###############################################################################################
#==============================================================================================
###############################################################################################

ic(model_list)
l_ml = len(model_list)
l_ml1 = len(model_list)+1

if challenge == 'challenge1':
    version = 'C1'   ##C1 is being used to massive trainings
    weights = 'imagenet'
    avgpool = True
    dropout = False
    optimizer = 'sgd'
    dataset_size = 20000
    vallim = 2000
    loss = 'binary_crossentropy'

testing = True
if testing:
    version = 'T3'  # VERSÃO PRA COLOCAR NAS PASTAS
    vallim = 50
    num_epochs = 5
    percent_data = 0.5
    dataset_size = 1000
    batch_size = 20
    
if ch2_testing:
    version = 'B1'

ic(testing)
code_data =[["learning rate:", learning_rate],
            ["classes:", num_classes],
            ["input_shape:", input_shape],["input_shape_vis:", input_shape_vis],
            ["augmented:", aug_data],["avg_pool?", avgpool],["loss:", loss],
            ["optimizer:", optimizer],
            ["dropout:", dropout],["use_alternative_model: ", use_alternative_model],
            ["preloading weights? ", preload_weights], 
            ["dataset_size:", dataset_size], ["valid:", vallim],
            ["percent_data:", percent_data], ["batch_size:", batch_size],
            ["num_epochs:", num_epochs], ["k_folds:", k_folds],
            ["weights:", weights],["testing?", testing],
            ['loss regularization?', loss_regularization],
            ["ch1_testing?", ch1_testing],["dirich_ensemble?", dirichlet],
            ["logistic_ensemble?", logistic],
            ["integrated-ensemble?", integrated_stacked],
            ['challenge:', challenge], ["VERSION:", version]]
ic(code_data)

with open('code_parameters_version_%s.csv' % version, 'w', newline='') as g:
    writer = csv.writer(g)
    writer.writerow(code_data)  #htop
##from joblib model_builderimport Parallel, delayed  #ssh c004
#Parallel(n_jobs=40)(delayed(func)(i) for i in range(200))
train_data_sizes = [15000.0]
#train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
#train_data_sizes = [190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
#train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]

#versions = [46, 47, 48, 49, 51, 52, 55, 54, 53, 50, 45, 44, 43, 42, 41, 40, 'B1', 'C2', 'C1', 56, 57, 58, 59, 61, 'T', 'T2']
#for version in versions:
#    print(' -- Current version: ')
#    for ix in model_list:
#        utils.delete_weights(ix, version)

if challenge == 'challenge1':
    train_data_sizes = [100, 480.0, 460.0, 440.0, 420.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2000.0, 1750.0, 1500.0, 1250.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]
    #train_data_sizes = [490.0, 480.0, 440.0, 410.0, 500.0, 390.0, 370.0, 360.0, 350.0, 340.0, 330.0, 310.0, 300.0, 290.0, 270.0, 260.0, 250.0, 210.0, 190.0, 180.0, 150.0, 140.0, 130.0, 110.0, 80.0, 70.0, 50.0, 40.0, 14000.0, 12000.0, 9000.0, 4000.0, 2500.0, 1500.0, 1400.0, 1300.0, 1250.0, 950.0, 800.0, 700.0, 600.0]
    #train_data_sizes = [14000]

if testing:
    train_data_sizes = [50.0]
ic(train_data_sizes)
#np.random.shuffle(train_data_sizes)

########################################################
# Checando dispositivos físicos disponíveis
ic(len(tf.config.experimental.list_physical_devices('GPU')))
ic(len(tf.config.experimental.list_physical_devices('CPU')))

# Verificando se o dataset está disponível ou será necessário fazer o download
if challenge == 'challenge1':
    x_data_original, y_data_original, index, channels = files_changer.data_downloader(dataset_size, version, input_shape)
elif challenge == 'challenge2':
    #from data_generator_challenge2 import DataGenerator
    from datagen_chal2 import DataGeneratorCh2
    #from datagen_chal2 import DataGeneratorCh2_ver_2
    #x_data_original, x_data_vis_orig, y_data_original, index, channels, test_inps, test_inps_v, test_outs = DataGeneratorCh2_ver_2(dataset_size, version, input_shape, input_shape_vis)
    x_data_original, x_data_vis_orig, y_data_original, index, channels, test_inps, test_inps_v, test_outs = DataGeneratorCh2(dataset_size, version, input_shape, input_shape_vis, using_238)
elif challenge == 'both':
    x_data_1, y_data_1, index, channels = files_changer.data_downloader(dataset_size, version, input_shape)
    from datagen_chal2 import DataGeneratorCh2
    x_data_2, y_data_2, index, channels, test_inps, test_inps_v, test_outs = DataGeneratorCh2(dataset_size, version, input_shape, input_shape_vis)

if ch2_testing:
    from datagen_chal2 import DataGeneratorCh2
    x_test_original, x_test_vis_orig, y_test_original, index, channels, test_inps, test_inps_v, test_outs = DataGeneratorCh2(10000, version, input_shape, input_shape_vis)


Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

# Contador de tempo Total
begin = time.perf_counter()
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

##########################################33
# Apangando dados de testes anteriores para executar o novo teste
ic('\n ** Cleaning up previous files...')
#files_changer.massive_cleaner(k_folds, num_epochs, 30000)
#files_changer.file_cleaner(k_folds, 39, input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 38, input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 'C2', input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 'C1', input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 43, input_shape, num_epochs, dataset_size)

if os.path.exists('RESULTS/ENSEMBLE_%s' % version):
    shutil.rmtree('RESULTS/ENSEMBLE_%s' % version)
    ic('\n ** Pasta ENSEMBLE removida')
    os.mkdir('RESULTS/ENSEMBLE_%s' % version)

# Loop para a execução de todos os conjuntos de DADOS DE TESTE
for mod in model_list:
    with open('code_data_version_%s_model_%s_aucs.csv' % (version, mod), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train_size', 'min', 'med', 'max'])
    with open('code_data_version_%s_model_%s_f1s.csv' % (version, mod), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train_size', 'min', 'med', 'max'])
for u in range(0, len(train_data_sizes)):
    begin_fold = time.perf_counter()
    train_size = train_data_sizes[u]
    print('\n\n\n ** NEW CICLE WITH %s TRAINING SAMPLES! **************************************************************************************************' % train_size)
    print('\n ** Cleaning up previous files and folders...')
    files_changer.file_remover(train_size, k_folds, version, model_list, num_epochs)

    print('\n ** Starting data preprocessing...')
    y_data = y_data_original
    x_data = x_data_original
    x_data_vis = x_data_vis_orig

    #y_data, x_data, y_test, x_test, y_val, x_val = utils.test_samples_balancer(y_data, x_data, vallim, train_size, percent_data, challenge)
    #y_data, x_data_vis, y_test, x_test_vis, y_val, x_val_vis = utils.test_samples_balancer(y_data, x_data_vis, vallim, train_size, percent_data, challenge)
    pad = np.zeros((x_data.shape[0],x_data.shape[1],x_data.shape[2],1), dtype="float32")
    x_data, x_val, y_data, y_val = train_test_split(np.concatenate([x_data[:,:,:,2:],pad,pad], axis=-1), y_data, test_size = 0.10, random_state = 7)
    x_data_vis, x_val_vis, y_data, y_val = train_test_split(x_data_vis, y_data_original, test_size = 0.10, random_state = 7)

    if ch2_testing:
        x_test = x_test_original
        y_test = y_test_original
    x_test = test_inps
    x_test_vis = test_inps_v
    y_test = to_categorical(test_outs, num_classes = 2)

    #from utils.dnn import split_data
    #x_data, y_data, x_val, y_val, x_test, y_test, idxs = split_data(x_data, y_data, validation_split = .1)

    ic(y_test.shape, y_data.shape, y_val.shape)
    ic(x_test.shape, x_data.shape, x_val.shape)
    ic(x_test_vis.shape, x_data_vis.shape, x_val_vis.shape)

    #############DISTRIBUTION GRAPH#########
    trainval_count = [np.count_nonzero(y_data == 1) + np.count_nonzero(y_val == 1),
                          np.count_nonzero(y_data == 0) + np.count_nonzero(y_val == 0)]
    test_count = [np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0)]

    # Gera o gráfico COMPLETO de distribuição dos dados
    print('\n ** Plotting Complete Distribution Graph')
    graphs.distribution_graph_complete(test_count, trainval_count, train_size, 'ensemble', 'ensemble', version)

    print('\n ** Converting data and list of indices into folds for cross-validation...')

    subset_size = int(len(y_data) / k_folds)
    folds = utils.load_data_kfold(k_folds, x_data, y_data)

    print('\n ** Starting network training... \n')

    start = time.perf_counter()
    roc_m = np.zeros((l_ml1,6), dtype = object)  ###0: AUC; 1:FPR; 2:TPR; 3:AUC2. SETTING DTYPE ALLOWS ANYTHING TO BE PUT INSIDE THAT ARRAY
    auc_m = np.zeros((l_ml1,3), dtype = object)  ###0:HIGHAUC; 1:LOWAUC; 2:MEDAUC
    f1s_m = np.zeros((l_ml1,4), dtype = object)  ###0: f1s; 1:FPR; 2:TPR; 3:AUC2.
    f1s_g = np.zeros((l_ml1,3), dtype = object)  ###0: 0:HIGHF1S; 1:LOWF1S; 2:MEDF1S;
    ic(roc_m.shape, auc_m.shape, f1s_m.shape)

    #y_test = to_categorical(y_test, num_classes=2)
    #print(" ** y_test: ", y_test)

    # Loop para a execução de todas as FOLDS
    for fold, (train_idx, val_idx) in enumerate(folds):

        print('\n\n\n **** New Fold ****')
        foldtimer = time.perf_counter()
        ic(fold, train_size)
        x_val_cv = x_val
        x_val_vis_cv = x_val_vis
        y_val_cv = y_val
        if train_size < 1600:
            print(' ** Using Original Cross-Val method')
            x_data_cv = x_data[val_idx]
            x_data_vis_cv = x_data_vis[val_idx]
            y_data_cv= y_data[val_idx]
        else:
            print(' ** Using Modified Cross-Val method')
            x_data_cv = x_data[train_idx]
            x_data_vis_cv = x_data_vis[train_idx]
            y_data_cv = y_data[train_idx]

            #############DISTRIBUTION GRAPH#########
        train_count = [np.count_nonzero(y_data_cv == 1), np.count_nonzero(y_data_cv == 0)]
        val_count = [np.count_nonzero(y_val_cv == 1), np.count_nonzero(y_val_cv == 0)]
        ic(train_count, val_count)

        # Gera o gráfico de distribuição dos dados por FOLD
        #index = utils.save_clue(x_data, y_data, dataset_size, version, 'generator', input_shape, 5, 5, index, channels)
        print('\n ** Plotting Fold Distribution Graph')
        graphs.distribution_graph_fold(train_count, val_count, train_size, fold, 'ensemble', 'ensemble', version)

        print('\n ** Converting vector classes to binary matrices...')
        y_data_cv = to_categorical(y_data_cv, num_classes=2)
        y_val_cv = to_categorical(y_val_cv, num_classes=2)

        ic(y_test.shape, y_data_cv.shape, y_val_cv.shape)
        ic(x_test.shape, x_data_cv.shape, x_val_cv.shape)
        ic(x_test_vis.shape, x_data_vis_cv.shape, x_val_vis_cv.shape)

        # Data Augmentation
        if aug_data:
            # TODO - TENTANDO AUTOMATIZAR O TIPO DE AUGMENTATION
            ic(aug_type)
            gen = ImageDataGenerator(aug_type)
        else:
            ic('\n ** No Augmented Data')
            gen = ImageDataGenerator()

        with tf.device('/GPU:0'):
            gen_flow = model_lib.gen_flow_for_two_inputs(x_data_vis_cv, x_data_cv, y_data_cv, batch_size, gen)
            #generator = gen.flow(x_data_cv, y_data_cv, batch_size=batch_size)
            ic(gen_flow)

        
            #" ** Generating models...")
            models = []
            ic(model_list)
            cn = 0
            for ix in model_list:
                ic(ix)
                model = model_lib.model_builder(ix, x_data, x_data_vis, weights, avgpool, dropout, preload_weights, loss_regularization, num_epochs, version)
                print(model.summary())
                callbacks = model_lib.compile_model(ix, model, optimizer, fold, version, learning_rate, loss, ch1_weights, batch_size)
                history = model_lib.fit_model(model, gen_flow, x_data_cv, x_data_vis_cv, batch_size, num_epochs, x_val_cv, x_val_vis_cv, y_val_cv, callbacks)
                ic('\n ** Training %s completed.' % model)
                ic(' ** Plotting Graphs')
                roc_m[cn,0], roc_m[cn,1], roc_m[cn,2], roc_m[cn,3], roc_m[cn,4], roc_m[cn,5], thres = graphs.roc_graph(ix, model, x_test, x_test_vis, y_test, ix, train_size, fold, version, ix)
                graphs.accurary_graph(history, num_epochs, train_size, fold, ix, ix, version)
                graphs.loss_graph(history, num_epochs, train_size, fold, ix, ix, version)
                #f1s_m[cn,0], f1s_m[cn,1], f1s_m[cn,2], f1s_m[cn,3] = graphs.fscore_graph(ix, model, x_test, y_test, ix, train_size, fold, version, ix, batch_size)
                #scores = model.evaluate(x_test, y_test, verbose=0)
                #print(' ** %s - Large CNN Error: %.2f%%' % (ix, (100 - scores[1] * 100)))
                cn = cn + 1
                models.append(model)
                files_changer.filemover(train_size, version, k_folds, model_list, dataset_size, input_shape, input_shape_vis)

            ###ROC CURVES GENERAL
            roc_m[cn,0], roc_m[cn,1], roc_m[cn,2], roc_m[cn,3], roc_m[cn,4], roc_m[cn,5], thres = graphs.roc_graphs_sec(rede, models, x_test, x_test_vis, y_test, model_list, train_size, fold, version, 'ensemble')
            #f1s_m[cn,0], f1s_m[cn,1], f1s_m[cn,2], f1s_m[cn,3] = graphs.fscore_graph_ensemble(model_list, models, x_test, y_test, train_size, fold, version, batch_size)

            #if integrated_stacked:
                #model_lib.integrated_stacked_model(models, x_test, x_test_vis, y_test, optimizer, train_size, model_list, version)

            fold_time_elapsed = (time.perf_counter() - foldtimer) / 60
            ic(fold_time_elapsed)
            K.clear_session() 

    elapsed = (time.perf_counter() - start) / 60
    ic(' ** %.3f TIME: %.3f minutes.' % (train_size, elapsed))

    ic(" ** Generating code_data for models.")
    for ind in range(len(models)):
        auc_m[ind,0], auc_m[ind,1], auc_m[ind,2], auc_m[ind,3], auc_m[ind,4], auc_m[ind,5] = graphs.ultimate_ROC(roc_m[ind,2], thres, roc_m[ind,0], roc_m[ind,1], roc_m[ind,5], roc_m[ind,3], roc_m[ind,4], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version)
        with open('code_data_version_%s_model_%s_aucs.csv' % (version, model_list[ind]), 'a', newline='') as f:
            writer = csv.writer(f)
            code_data = [train_size, auc_m[ind,0], auc_m[ind,1], auc_m[ind,2], auc_m[ind,3], auc_m[ind,4], auc_m[ind,5]]
            writer.writerow(code_data)
        #f1s_g[ind,0], f1s_g[ind,1], f1s_g[ind,2] = graphs.ultimate_fscore(f1s_m[ind,2], f1s_m[ind,3], thres, f1s_m[ind,0], f1s_m[ind,1], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version, batch_size)
        #with open('code_data_version_%s_model_%s_f1s.csv' % (version, model_list[ind]), 'a', newline='') as f:
            #writer = csv.writer(f)
            #code_data = [train_size, f1s_g[ind,0], f1s_g[ind,1], f1s_g[ind,2]]
            #writer.writerow(code_data)

    files_changer.filemover(train_size, version, k_folds, model_list, dataset_size, input_shape, input_shape_vis)
    auc_m[l_ml,0], auc_m[l_ml,1], auc_m[l_ml,2], auc_m[ind,3], auc_m[ind,4], auc_m[ind,5] = graphs.ultimate_ROC(roc_m[(l_ml),2], thres, roc_m[(l_ml),0], roc_m[(l_ml),1], roc_m[ind,5], roc_m[ind,3], roc_m[ind,4], 'ensemble', 'ensemble', k_folds, train_size, 'ensemble', version)
    #f1s_g[l_ml,0], f1s_g[l_ml,1], f1s_g[l_ml,2] = graphs.ultimate_fscore(f1s_m[l_ml,2], f1s_m[l_ml,3], thres, f1s_m[l_ml,0], f1s_m[l_ml,1], 'ensemble', 'ensemble', k_folds, train_size, 'ensemble', version)

    k_back.clear_session()

    time_fold = (time.perf_counter() - begin_fold) / (60 * 60)
    ic('\n Ciclo ', u, ' concluido em: ', time_fold, ' horas.')

files_changer.last_mover(version, model_list, dataset_size, num_epochs, input_shape)

time_total = (time.perf_counter() - begin) / (60 * 60)
ic('\n ** Mission accomplished in {} hours.'. format(time_total))
ic('\n ** FINISHED! ************************')






