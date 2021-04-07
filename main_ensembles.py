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
from keras.applications.resnet50 import ResNet50
# Todo - Testar a resnet 50 V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2

from utils import files_changer, utils, graphs
import model_lib

warnings.filterwarnings('ignore')

print('\n ## Tensorflow version:')
print(tf.__version__)
print(' ## Is GPU available?')
print(tf.test.is_gpu_available())

Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

cwd = os.getcwd()
print(cwd)
print(cwd)
#PARAMETERS
version = 51  # EXECUTION VERSION
num_classes = 2  # NUMBER OF OUTPUT CLASSES
rede = 'ensemble'     ##OPTIONS: 'resnet', 'ensemble' or 'effnet'
weights = 'imagenet'   #'imagenet' or None
preload_weights = False
ch2_testing = False
ch1_weights = False
aug_data = True     ##DO YOU WANT SOME DATA AUGMENTATION?
aug_type = ['rotation_range=90, horizontal_flip=True, vertical_flip=True']
learning_rate = 0.01
optimizer = 'sgd'  ##CHOOSE 'sgd' or 'adam' or 'nadam'
avgpool = True
dropout = False
loss = 'binary_crossentropy'
num_epochs = 50  # NUMBER OF EPOCHS
batch_size = 64  # Tamanho do Batch
k_folds = 10  # NÚMERO DE FOLDS
percent_data = 1.0  # Porcentagem de Dados DE TESTE a ser usado - TODO - Definir como será utilizado
vallim = 3000  # Quantidade de dados de validação
challenge = 'challenge1'
##about types of ensembles used
dirichlet = False
logistic = False
integrated_stacked = False
dataset_size = 20000
input_shape = 101
classes = ['lens', 'not-lens']
model_list = ['resnet50', 'effnet_B2']
#model_list = ['resnet50', 'resnet101', 'resnet152', 'effnet_B0', 'effnet_B1', 'effnet_B2', 'effnet_B3', 'effnet_B4', 'effnet_B5', 'effnet_B6', 'effnet_B7', 'inceptionV2', 'inceptionV3', 'xception']
print('\n ** Going to loop through models: ', model_list)
l_ml = len(model_list)
l_ml1 = len(model_list)+1

if challenge == 'challenge1':
    version = 'C2'   ##C1 is being used to massive trainings
    preload_weights = False
    weights = 'imagenet'
    avgpool = True
    dropout = False
    dataset_size = 20000
    vallim = 2000

testing = False
if testing:
    version = 'T'  # VERSÃO PRA COLOCAR NAS PASTAS
    vallim = 50
    num_epochs = 5
    percent_data = 0.1
    dataset_size = 2000

print("\n ** Are we performing tests? :", testing)
print("\n ** Chosen parameters:")
code_data =[["learning rate:", learning_rate],
            ["classes:", num_classes],
            ["input_shape:", input_shape],
            ["augmented:", aug_data],["avg_pool?", avgpool],["loss:", loss],
            ["dropout:", dropout],
            ["dataset_size:", dataset_size], ["valid:", vallim],
            ["percent_data:", percent_data], ["batch_size:", batch_size],
            ["num_epochs:", num_epochs], ["k_folds:", k_folds],
            ["weights:", weights],["testing?", testing],
            ["dirich_ensemble?", dirichlet],
            ["logistic_ensemble?", logistic],
            ["integrated-ensemble?", integrated_stacked],
            ['challenge:', challenge], ["VERSION:", version]]
print(code_data)

with open('code_parameters_version_%s.csv' % version, 'w', newline='') as g:
    writer = csv.writer(g)
    writer.writerow(code_data)  #htop
##from joblib import Parallel, delayed  #ssh c004
#Parallel(n_jobs=40)(delayed(func)(i) for i in range(200))
train_data_sizes = [800.0]
#train_data_sizes = [1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]
#16000.0, 14000.0, 12000.0, 10000.0, 7500.0, 5000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0
#train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
#train_data_sizes = [190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
#train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]

if challenge == 'challenge1':
    train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]
    #train_data_sizes = [10000.0]

if testing:
    train_data_sizes = [100.0]
print('\n ** train_data_sizes: ', train_data_sizes)
np.random.shuffle(train_data_sizes)

########################################################
# Checando dispositivos físicos disponíveis
print('\n\n ## Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
print('\n ## Num CPUs Available: ', len(tf.config.experimental.list_physical_devices('CPU')))

# Verificando se o dataset está disponível ou será necessário fazer o download
print('\n ** Verifying data...')
if challenge == 'challenge1':
    x_data_original, y_data_original, index, channels = files_changer.data_downloader(dataset_size, version, input_shape)
else:
    from data_generator_challenge2 import DataGenerator
    from datagen_chal2 import DataGeneratorCh2
    #x_data_original, y_data_original, index, channels = DataGenerator(dataset_size, version, input_shape)
    x_data_original, y_data_original, index, channels = DataGeneratorCh2(dataset_size, version, input_shape)
if ch2_testing:
    from datagen_chal2 import DataGeneratorCh2
    x_test_original, y_test_original, index, channels = DataGeneratorCh2(10000, version, input_shape)
Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

# Contador de tempo Total
begin = time.perf_counter()
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

##########################################33
# Apangando dados de testes anteriores para executar o novo teste
print('\n ** Cleaning up previous files...')
#files_changer.massive_cleaner(k_folds, num_epochs, 30000)
#files_changer.file_cleaner(k_folds, 39, input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 38, input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 'C2', input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 'C1', input_shape, num_epochs, dataset_size)
#files_changer.file_cleaner(k_folds, 43, input_shape, num_epochs, dataset_size)

if os.path.exists('RESULTS/ENSEMBLE_%s' % version):
    shutil.rmtree('RESULTS/ENSEMBLE_%s' % version)
    print('\n ** Pasta ENSEMBLE removida')
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

    #print('\n ** Randomizing y_data and x_data...')
    #y_data = np.array(y_data)
    #x_data = np.array(x_data)
    #i = np.arange(y_data.shape[0])
    #np.random.shuffle(i)
    #y_data = y_data[i]
    #x_data = x_data[i]

    #print('\n ** y_data shape: ', y_data.shape, ' ** Total dataset size: ', len(y_data), 'objects.')
    #print('\n ** Balancing number of samples on each class for train+val sets with %s samples...' % train_size)
    y_data, x_data, y_test, x_test, y_val, x_val = utils.test_samples_balancer(y_data, x_data, vallim, train_size, percent_data, challenge)
    if ch2_testing:
        x_test = x_test_original
        y_test = y_test_original
    y_test = to_categorical(y_test)

    #from utils.dnn import split_data
    #x_data, y_data, x_val, y_val, x_test, y_test, idxs = split_data(x_data, y_data, validation_split = .1)

    print('\n ** y_data arranged with format: \n ** y_test:   ', y_test.shape, '\n ** y_data:  ', y_data.shape, '\n ** y_val:  ', y_val.shape)
    print('\n ** x_data splitted with format: \n ** x_test:   ', x_test.shape, '\n ** x_data:  ', x_data.shape, '\n ** x_val:  ', x_val.shape)

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
    roc_m = np.zeros((l_ml1,4), dtype = object)  ###0: AUC; 1:FPR; 2:TPR; 3:AUC2. SETTING DTYPE ALLOWS ANYTHING TO BE PUT INSIDE THAT ARRAY
    auc_m = np.zeros((l_ml1,3), dtype = object)  ###0:HIGHAUC; 1:LOWAUC; 2:MEDAUC
    f1s_m = np.zeros((l_ml1,4), dtype = object)  ###0: f1s; 1:FPR; 2:TPR; 3:AUC2.
    f1s_g = np.zeros((l_ml1,3), dtype = object)  ###0: 0:HIGHF1S; 1:LOWF1S; 2:MEDF1S;
    print(' ** roc_m: %s, auc_m: %s, f1s_m: %s' % (roc_m.shape, auc_m.shape, f1s_m.shape))

    print(" ** y_test: ", y_test)
    #y_test = to_categorical(y_test, num_classes=2)
    #print(" ** y_test: ", y_test)

    # Loop para a execução de todas as FOLDS
    for fold, (train_idx, val_idx) in enumerate(folds):

        print('\n\n\n **** New Fold ****')
        foldtimer = time.perf_counter()
        print('\n ** Fold: %s with %s training samples' % (fold, train_size))
        x_val_cv = x_val
        y_val_cv = y_val
        if train_size < 1600:
            print(' ** Using Original Cross-Val method')
            x_data_cv = x_data[val_idx]
            y_data_cv= y_data[val_idx]
        else:
            print(' ** Using Modified Cross-Val method')
            x_data_cv = x_data[train_idx]
            y_data_cv = y_data[train_idx]

        #############DISTRIBUTION GRAPH#########
        train_count = [np.count_nonzero(y_data_cv == 1), np.count_nonzero(y_data_cv == 0)]
        val_count = [np.count_nonzero(y_val_cv == 1), np.count_nonzero(y_val_cv == 0)]

        # Gera o gráfico de distribuição dos dados por FOLD
        index = utils.save_clue(x_data, y_data, dataset_size, version, 'generator', input_shape, 5, 5, index, channels)
        print('\n ** Plotting Fold Distribution Graph')
        graphs.distribution_graph_fold(train_count, val_count, train_size, fold, 'ensemble', 'ensemble', version)

        print('\n ** Converting vector classes to binary matrices...')
        y_data_cv_antes = y_data_cv
        y_val_cv_antes = y_val_cv
        y_data_cv = to_categorical(y_data_cv, num_classes=2)
        y_val_cv = to_categorical(y_val_cv, num_classes=2)

        # Data Augmentation
        if aug_data:
            # TODO - TENTANDO AUTOMATIZAR O TIPO DE AUGMENTATION
            print('\n ** Augmented Data: {}'. format(aug_type))
            gen = ImageDataGenerator(aug_type)
        else:
            print('\n ** Sem Augmented Data')
            gen = ImageDataGenerator()
        generator = gen.flow(x_data_cv, y_data_cv, batch_size=batch_size)
        print(generator)

        with tf.device('/GPU:0'):
        # with tf.device('/GPU:1'):
        # with tf.device('/GPU:3'):

        # Gerando a rede
            print(" ** Generating models...")
            models = []
            print(" ** from: ", model_list)
            cn = 0
            for ix in model_list:
                print('\n ** Current model: ', ix)
                model = model_lib.model_builder(ix, x_data, weights, avgpool, dropout, preload_weights)
                print('\n ** Model Summary: \n', model.summary())
                print('\n ** Training %s.' % ix)
                callbacks = model_lib.compile_model(ix, model, optimizer, fold, version, learning_rate, loss, ch1_weights)
                #print(model, gen, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks)
                history = model_lib.fit_model(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks)
                print('\n ** Training %s completed.' % model)
                print(' ** Plotting %s Graphs' % model)
                graphs.accurary_graph(history, num_epochs, train_size, fold, ix, ix, version)
                graphs.loss_graph(history, num_epochs, train_size, fold, ix, ix, version)
                roc_m[cn,0], roc_m[cn,1], roc_m[cn,2], roc_m[cn,3], thres = graphs.roc_graph(ix, model, x_test, y_test, ix, train_size, fold, version, ix)
                f1s_m[cn,0], f1s_m[cn,1], f1s_m[cn,2], f1s_m[cn,3] = graphs.fscore_graph(ix, model, x_test, y_test, ix, train_size, fold, version, ix)
                #scores = model.evaluate(x_test, y_test, verbose=0)
                #print(' ** %s - Large CNN Error: %.2f%%' % (ix, (100 - scores[1] * 100)))
                cn = cn + 1
                models.append(model)
                utils.delete_weights(ix, version)

            ###ROC CURVES ENSEMBLES
            roc_m[cn,0], roc_m[cn,1], roc_m[cn,2], roc_m[cn,3], thres = graphs.roc_graphs_sec(rede, models, x_test, y_test, model_list, train_size, fold, version, 'ensembles')
            f1s_m[cn,0], f1s_m[cn,1], f1s_m[cn,2], f1s_m[cn,3] = graphs.fscore_graph_ensemble(model_list, models, x_test, y_test, train_size, fold, version)

            elaps = (time.perf_counter() - foldtimer) / 60
            print('\n ** Fold TIME: %.3f minutes.' % elaps)
            K.clear_session() 

    # CLOSE Loop para a execução de todas as FOLDS
    print('\n ** Training and evaluation complete.')
    elapsed = (time.perf_counter() - start) / 60
    print(' ** %.3f TIME: %.3f minutes.' % (train_size, elapsed))

    print(" ** Generating code_data for models.")
    for ind in range(len(models)):
        auc_m[ind,0], auc_m[ind,1], auc_m[ind,2] = graphs.ultimate_ROC(roc_m[ind,2], roc_m[ind,3], thres, roc_m[ind,0], roc_m[ind,1], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version)
        with open('code_data_version_%s_model_%s_aucs.csv' % (version, model_list[ind]), 'a', newline='') as f:
            writer = csv.writer(f)
            code_data = [train_size, auc_m[ind,0], auc_m[ind,1], auc_m[ind,2]]
            writer.writerow(code_data)
        f1s_g[ind,0], f1s_g[ind,1], f1s_g[ind,2] = graphs.ultimate_fscore(f1s_m[ind,2], f1s_m[ind,3], thres, f1s_m[ind,0], f1s_m[ind,1], model_list[ind], model_list[ind], k_folds, train_size, model_list[ind], version)
        with open('code_data_version_%s_model_%s_f1s.csv' % (version, model_list[ind]), 'a', newline='') as f:
            writer = csv.writer(f)
            code_data = [train_size, f1s_g[ind,0], f1s_g[ind,1], f1s_g[ind,2]]
            writer.writerow(code_data)

    files_changer.filemover(train_size, version, k_folds, model_list, num_epochs)
    auc_m[l_ml,0], auc_m[l_ml,1], auc_m[l_ml,2] = graphs.ultimate_ROC(roc_m[(l_ml),2], roc_m[(l_ml),3], thres, roc_m[(l_ml),0], roc_m[(l_ml),1], 'ensemble', 'ensemble', k_folds, train_size, 'ensemble', version)
    f1s_g[l_ml,0], f1s_g[l_ml,1], f1s_g[l_ml,2] = graphs.ultimate_fscore(f1s_m[l_ml,2], f1s_m[l_ml,3], thres, f1s_m[l_ml,0], f1s_m[l_ml,1], 'ensemble', 'ensemble', k_folds, train_size, 'ensemble', version)

    k_back.clear_session()

    time_fold = (time.perf_counter() - begin_fold) / (60 * 60)
    print('\n Ciclo ', u, ' concluido em: ', time_fold, ' horas.')

files_changer.last_mover(version, model_list, dataset_size, num_epochs, input_shape)

time_total = (time.perf_counter() - begin) / (60 * 60)
print('\n ** Mission accomplished in {} hours.'. format(time_total))
print('\n ** FINISHED! ************************')






