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
from sklearn import metrics
from deepstack.base import KerasMember
from deepstack.ensemble import DirichletEnsemble, StackEnsemble
from sklearn.ensemble import RandomForestRegressor, StackingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from effnet import get_model_effnet, compile_model_effnet, fit_model_effnet
from ensemble import stacked_logisticregression_ensemble, integrated_stacked_model
from resnet import get_model_resnet, compile_model_resnet, fit_model_resnet
# from utils import utils, graphs, files_changer
# from utils.utils import select_optimizer
# import utils
from utils import files_changer, utils, graphs

warnings.filterwarnings('ignore')

print('\n ## Tensorflow version:')
print(tf.__version__)

print(' ## Is GPU available?')
print(tf.test.is_gpu_available())

# Path('C:/Users/Teletrabalho/Documents/Pessoal/resultados').parent
# os.chdir('C:/Users/Teletrabalho/Documents/Pessoal/resultados')

# Path('/home/hdd2T/icaroDados').parent
# os.chdir('/home/hdd2T/icaroDados')

# GPU NEW

# IDGPUs = '0,1'; os.environ['CUDA_VISIBLE_DEVICES'] = 'PCI_BUS_ID'; os.environ["CUDA_VISIBLE_DEVICES"] = IDGPUs


#Path('/home/dados2T/icaroDados/resnetLens').parent
#os.chdir('/home/dados2T/icaroDados/resnetLens')
Path('/home/kayque/LENSLOAD/').parent
os.chdir('/home/kayque/LENSLOAD/')

# Parâmetros
version = 26  # VERSÃO PRA COLOCAR NAS PASTAS
num_classes = 2  # Número de classes de saída da rede.

# Todo - Variáveis de configuração das redes *****************

# Define qual resnet deve ser utilizada (profundidade)
resnet_depth = 50
# resnet_depth = 101
# resnet_depth = 152

# effnet_version = 'B0'
# effnet_version = 'B1'
effnet_version = 'B2'
# effnet_version = 'B3'
# effnet_version = 'B4'
# effnet_version = 'B5'
# effnet_version = 'B6'
# effnet_version = 'B7'

# Qual rede utilizar
# rede = 'resnet'
#rede = 'effnet'
rede = 'ensemble'

# Pesos imgnet ou None
# weights = 'none'
# weights = None
weights = 'imagenet'

# Augmented Data
#aug_data = True
aug_data = True
aug_type = ['rotation_range=90, horizontal_flip=True, vertical_flip=True']

# lr = 0.01
learning_rate = 0.01
# Otimizador
#optimizer = 'sgd'
optimizer = 'sgd' #SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = 'adam'
num_epochs = 50  # Épocas
# num_epochs = 10  # Épocas
batch_size = 64  # Tamanho do Batch
#batch_size = 32  # Tamanho do Batch
k_folds = 10  # NÚMERO DE FOLDS
# k_folds = 5  # NÚMERO DE FOLDS
percent_data = 1.0  # Porcentagem de Dados a ser usado - Todo - Definir como será utilizado
vallim = 2000  # Quantidade de dados de validação
dirichlet = False
logistic = False
integrated_stacked = False

# Reduzir tamanho dos dados de teste para execução mais rápida da rede, possibilitando o debug de erros.
# test_peq = True
test_peq = False
test_size = 50
testing = False
if testing:
    version = 'T'  #VERSÃO PRA COLOCAR NAS PASTAS

print("\n ** Are we performing tests? :", testing)
if testing:
    vallim = 50
    num_epochs = 20
    percent_data = 0.1


#train_data_sizes = [16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]
#16000.0, 14000.0, 12000.0, 10000.0, 7500.0, 5000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0
#train_data_sizes = [np.array(np.linspace(500.0, 20.0, 49))]
#train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0,
#                    390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0,
#                    200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0]
#                    90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
#train_data_sizes = [190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0,
#                    200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
train_data_sizes = [490.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0, 420.0, 410.0, 500.0, 390.0, 380.0, 370.0, 360.0, 350.0, 340.0, 330.0, 320.0, 310.0, 300.0, 200.0, 290.0, 280.0, 270.0, 260.0, 250.0, 240.0, 230.0, 220.0, 210.0, 400.0, 190.0, 180.0, 170.0, 160.0, 150.0, 140.0, 130.0, 120.0, 110.0, 100.0, 200.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 16000.0, 14000.0, 12000.0, 10000.0, 9000.0, 8000.0, 7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2500.0, 2250.0, 2000.0, 1750.0, 1500.0, 1400.0, 1300.0, 1200.0, 1100.0, 1000.0, 950.0, 900.0, 850.0, 800.0, 750.0, 700.0, 650.0, 600.0, 550.0]

# Tamanho padrão dos testes
# train_data_sizes = [np.array([600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50])]

print('\n ** train_data_sizes: ', train_data_sizes)
np.random.shuffle(train_data_sizes)

# Todo FIM *****************
dataset_size = 20000
input_shape = 101
TR = 16000
classes = ['lens', 'not-lens']

print("\n ** Chosen parameters:")

code_data =[["learning rate", learning_rate],
            ["classes", num_classes],
            ["input_shape", input_shape],
            ["augmented", aug_data],
            ["dataset_size", dataset_size], ["TR", TR], ["valid", vallim],
            ["percent_data", percent_data], ["batch_size", batch_size],
            ["num_epochs", num_epochs], ["k_folds", k_folds],
            ["rede", rede], ["weights", weights],["resnet_depth", resnet_depth],
            ["effnet_version", effnet_version],
            ["dirich_ensemble?", dirichlet],
            ["logistic_ensemble?", logistic],
            ["integrated-ensemble?", integrated_stacked],
            ["VERSION", version]]
print(code_data)

with open('code_parameters_version_%s.csv' % version, 'w', newline='') as g:
    writer = csv.writer(g)
    writer.writerow(code_data)

########################################################
# Checando dispositivos físicos disponíveis
print('\n\n ## Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))
# print('\n\n ## Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
print('\n ## Num CPUs Available: ', len(tf.config.experimental.list_physical_devices('CPU')))
# print('\n ## Num CPUs Available: ', len(tf.config.list_physical_devices('CPU')))

##########################################33
# Apangando dados de testes anteriores para executar o novo teste

# Verificando se o dataset está disponível ou será necessário fazer o download
print('\n ** Verifying data...')
files_changer.data_downloader()

# Carregando o y_data
print('\n ** Reading data from y_data20000fits.csv...')
# LOAD Y_DATA
PATH = os.getcwd()
var = PATH + '/' + 'lensdata/'
y_batch = os.listdir(var)

# Contador de tempo Total
begin = time.perf_counter()
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'

# Variáveis que armazenam as predições da rede Máx, mín e médio
hlow_auc, hhigh_auc, hmed_auc, med_dir_AUC, low_dir_AUC, hig_dir_AUC, med_int_AUC, low_int_AUC, hig_int_AUC = ([] for i in range(9))

# TODO - mudar o nome dessas variáveis
# Auc_min, Auc_max, Auc_med = ([] for i in range(3))

# Cria uma variável com o nome a ser exibido no Título dos Gráficos e nos arquivos criados
title_graph_rede, name_file_rede = utils.define_nomes_redes(rede, resnet_depth, effnet_version)
title_graph_rede_resnet, name_file_rede_resnet, title_graph_rede_effnet, name_file_rede_effnet = utils.nomes_extra_ensenble()

print('\n ** Cleaning up previous files...')
#files_changer.massive_cleaner(k_folds, name_file_rede, num_epochs, dataset_size)
files_changer.file_cleaner(k_folds, version, name_file_rede, num_epochs, dataset_size)
if os.path.exists('RESULTS/ENSEMBLE_%s' % version):
    shutil.rmtree('RESULTS/ENSEMBLE_%s' % version)
    print('\n ** Pasta ENSEMBLE removida')
    os.mkdir('RESULTS/ENSEMBLE_%s' % version)

# Loop para a execução de todos os conjuntos de DADOS DE TESTE
with open('code_data_version_%s.csv' % version, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['train_size', 'min_RN_AUC', 'RN_AUC', 'max_RN_AUC',
        'min_EN_AUC', 'EN_AUC', 'max_EN_AUC', 'min_dir_AUC', 'dir_AUC',
        'max_dir_AUC', 'min_log_AUC', 'log_AUC', 'max_log_AUC', 'min_int_AUC',
        'int_AUC', 'max_int_AUC', 'min_Med_AUC', 'Med_AUC', 'max_Med_AUC',
        'prec_all', 'rec_all', 'f_1_score_all', 'f_100_score_all',
        'prec_all_rn', 'rec_all_rn', 'f_1_score_all_rn', 'f_100_score_all_rn',
        'prec_all_ef', 'rec_all_ef', 'f_1_score_all_ef', 'f_100_score_all_ef',])
    for u in range(0, len(train_data_sizes)):
        # Contador de tempo por Fold
        begin_fold = time.perf_counter()

        # Função que altera o batch_size ao longo da execução
        # batch_size = change_batch_size(u)
        # print('\n batch_size: ', batch_size)

        train_size = train_data_sizes[u]
        print('\n ** train_size: ', train_size)

        # # Porcentagem de Dados a ser usado - Todo - Definir como será utilizado
        # vallim_teste = int(train_size * percent_data)
        # print('\n vallim_teste: ', vallim_teste, '-- total data: ', train_size, '-- percent_data: ', percent_data)
        #
        print(
            '\n\n\n ** NEW CICLE WITH %s TRAINING SAMPLES! **************************************************************************************************' % train_size)
        ######################################################
        print('\n ** Cleaning up previous files and folders...')
        files_changer.file_remover(train_size, k_folds, version, name_file_rede, num_epochs)

        # Processando os dados
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

        print('\n ** Randomizing y_data and x_data...')
        ind = np.arange(y_data.shape[0])
        np.random.shuffle(ind)
        y_data = y_data[ind]
        x_data = x_data[ind]

        print('\n ** y_data shape: ', y_data.shape)
        print(' ** Total dataset size: ', y_size, 'objects.')

        # Separando os dados de forma balanceada
        print('\n ** Balancing number of samples on each class for train+val sets with %s samples...' % train_size)
        y_data, x_data, y_test, x_test, y_val, x_val = utils.test_samples_balancer(y_data, x_data, vallim, train_size, test_peq, test_size, 'challenge1')

        print('\n ** y_data arranged with format:')
        print(' ** y_test:   ', y_test.shape)
        print(' ** y_data:  ', y_data.shape)
        print(' ** y_val:  ', y_val.shape)

        print('\n ** x_data splitted with format:')
        print(' ** x_test:   ', x_test.shape)
        print(' ** x_data:  ', x_data.shape)
        print(' ** x_val:  ', x_val.shape)

        y_size = len(y_data)
        y_tsize = len(y_test)
        x_size = len(x_data)

        if test_peq:
            # percent_data = AMOUNT OF THE DATASET USED (< 1 FOR TESTING)
            # TODO - está funcional, definir como será usado
            y_val_menor = y_val[: int(len(y_val) * percent_data), :]
            x_val_menor = x_val[: int(len(x_val) * percent_data), :]
            y_test_menor = y_test[: int(len(y_test) * percent_data), :]
            x_test_menor = x_test[: int(len(x_test) * percent_data), :]

            # print('\n ** Dados Reduzidos Teste:')
            # print('\n')
            # print(' ** y_val_menor:   ', y_val_menor.shape)
            # print(' ** x_val_menor:  ', x_val_menor.shape)
            # print(' ** y_test_menor:  ', y_test_menor.shape)
            # print(' ** x_test_menor:  ', x_test_menor.shape)
        ##############

        #############DISTRIBUTION GRAPH#########
        trainval_count = [np.count_nonzero(y_data == 1) + np.count_nonzero(y_val == 1),
                          np.count_nonzero(y_data == 0) + np.count_nonzero(y_val == 0)]
        test_count = [np.count_nonzero(y_test == 1), np.count_nonzero(y_test == 0)]

        # Gera o gráfico COMPLETO de distribuição dos dados
        print('\n ** Plotting Complete Distribution Graph')
        graphs.distribution_graph_complete(test_count, trainval_count, train_size, name_file_rede, title_graph_rede, version)

        print('\n ** Converting data and list of indices into folds for cross-validation...')

        subset_size = int(y_size / k_folds)
        folds = utils.load_data_kfold(k_folds, x_data, y_data)

        print('\n ** Starting network training... \n')

        start = time.perf_counter()
        if rede == 'ensemble':
            fpr_all, tpr_all, auc_all, lauc = ([] for i in range(4))
            acc0_res, loss0_res, val_acc0_res, val_loss0_res = ([] for i in range(4))
            acc0_eff, loss0_eff, val_acc0_eff, val_loss0_eff = ([] for i in range(4))
            fpr_all_rn, tpr_all_rn, auc_all_rn, lauc_rn = ([] for i in range(4))
            fpr_all_ef, tpr_all_ef, auc_all_ef, lauc_ef = ([] for i in range(4))
            dir_AUC_all, log_AUC_all, integs_AUC_all = ([] for i in range(3))
            prec_all, rec_all, f_1_score_all, f_100_score = ([] for i in range(4))
            prec_all_rn, rec_all_rn, f_1_score_all_rn, f_100_score_all_rn = ([] for i in range(4))
            prec_all_ef, rec_all_ef, f_1_score_all_ef, f_100_score_all_ef = ([] for i in range(4))
        else:
            fpr_all, tpr_all, auc_all, acc0, loss0, val_acc0, val_loss0, lauc = ([] for i in range(8))

        y_test = to_categorical(y_test, num_classes=2)

        # Loop para a execução de todas as FOLDS
        for fold, (train_idx, val_idx) in enumerate(folds):

            print('\n\n\n **** New Fold ****')
            foldtimer = time.perf_counter()
            print('\n ** Fold: %s with %s training samples' % (fold, train_size))
            x_val_cv = x_val
            y_val_cv = y_val
            #x_data_cv = x_data[val_idx]
            #y_data_cv = y_data[val_idx]
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
            print('\n ** Plotting Fold Distribution Graph')
            graphs.distribution_graph_fold(train_count, val_count, train_size, fold, name_file_rede, title_graph_rede, version)

            print('\n ** Converting vector classes to binary matrices...')
            y_data_cv_antes = y_data_cv
            y_val_cv_antes = y_val_cv
            y_data_cv = to_categorical(y_data_cv, num_classes=2)
            y_val_cv = to_categorical(y_val_cv, num_classes=2)

            # Gerando a rede
            if rede == 'resnet':
                model = get_model_resnet(x_data, weights, resnet_depth)
                print('\n ** Model Resnet Summary: \n', model.summary())

            elif rede == 'effnet':
                model = get_model_effnet(x_data, weights, effnet_version)
                print('\n ** Model Efficient Net Summary: \n', model.summary())

            else:
                model_resnet = get_model_resnet(x_data, weights, resnet_depth)
                model_effnet = get_model_effnet(x_data, weights, effnet_version)

                # model = get_model_ensemble(x_data, weights, resnet_depth, effnet_version)
                # model_resnet = model[0]
                # model_effnet = model[1]

                # model_resnet, model_effnet = get_model_ensemble(x_data, weights, resnet_depth, effnet_version)

                # TODO INVERTENDO A ORDEM DE TREINAMENTO DAS REDES
                # model_effnet = model[0]
                # model_resnet = model[1]
                # print('\n ** Model Resnet Summary: \n', model[0].summary())
                # print('\n ** Model Efficient Net Summary: \n', model[1].summary())

            print('\n ** Compiling model...')

            # Selecionando o otimizador da rede
            if rede == 'ensemble':
                learning_rate_res = learning_rate
                learning_rate_efn = learning_rate
                opt_res = utils.select_optimizer(optimizer, learning_rate_res)
                opt_efn = utils.select_optimizer(optimizer, learning_rate_efn)

            else:
                opt = utils.select_optimizer(optimizer, learning_rate)

            with tf.device('/GPU:0'):
            # with tf.device('/GPU:1'):
            # with tf.device('/GPU:3'):

                # Data Augmentation
                if aug_data:

                    # TODO - TENTANDO AUTOMATIZAR O TIPO DE AUGMENTATION
                    print('\n ** Augmented Data: {}'. format(aug_type))
                    gen = ImageDataGenerator(aug_type)
                else:
                    print('\n ** Sem Augmented Data')
                    gen = ImageDataGenerator()

                generator = gen.flow(x_data_cv, y_data_cv, batch_size=batch_size)

                if rede == 'resnet':
                    # Compile and Fit Model
                    callbacks = compile_model_resnet(name_file_rede, model, opt, fold, version)
                    history = fit_model_resnet(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks)
                    print('\n ** Training completed.')

                    # Plot Acc Graph and Loss Graph
                    # nome_rede = 'Resnet' + str(resnet_depth)
                    # n_rede = rede + str(resnet_depth)
                    print('\n ** Plotting Resnet Graphs')
                    graphs.accurary_graph(history, num_epochs, train_size, fold, name_file_rede, title_graph_rede, version)
                    graphs.loss_graph(history, num_epochs, train_size, fold, name_file_rede, title_graph_rede, version)
                    auc_all, fpr_all, tpr_all, lauc = graphs.roc_graph(rede, model, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede, auc_all, fpr_all, tpr_all, lauc)
                    prec_all, rec_all, f_1_score_all, f_100_score_all = graphs.fscore_graph(rede, model, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede, prec_all, rec_all, f_1_score_all, f_100_score_all)

                    acc0, val_acc0, loss0, val_loss0 = utils.acc_score(acc0, history, val_acc0, loss0, val_loss0)
                    scores = model.evaluate(x_test, y_test, verbose=0)
                    print(' ** Large CNN Error: %.2f%%' % (100 - scores[1] * 100))

                elif rede == 'effnet':
                    # Compile and Fit Model
                    callbacks = compile_model_effnet(name_file_rede, model, opt, fold, version)
                    history = fit_model_effnet(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks)
                    print('\n ** Training completed.')

                    print('\n ** Plotting Efficient Net Graphs')
                    graphs.accurary_graph(history, num_epochs, train_size, fold, name_file_rede, title_graph_rede, version)
                    graphs.loss_graph(history, num_epochs, train_size, fold, name_file_rede, title_graph_rede, version)
                    auc_all, fpr_all, tpr_all, lauc = graphs.roc_graph(rede, model, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede, auc_all, fpr_all, tpr_all, lauc)
                    prec_all, rec_all, f_1_score_all, f_100_score_all = graphs.fscore_graph(rede, model, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede, prec_all, rec_all, f_1_score_all, f_100_score_all)

                    acc0, val_acc0, loss0, val_loss0 = utils.acc_score(acc0, history, val_acc0, loss0, val_loss0)
                    scores = model.evaluate(x_test, y_test, verbose=0)
                    print(' ** Large CNN Error: %.2f%%' % (100 - scores[1] * 100))

                else:  # Ensemble
                    print('\n ** Training ResNet.')
                    callbacks_resnet = compile_model_resnet(name_file_rede, model_resnet, opt_res, fold, version)
                    history_resnet = fit_model_resnet(model_resnet, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks_resnet)
                    print('\n ** Training ResNet completed.')
                    print(' ** Plotting ResNet Graphs')
                    graphs.accurary_graph(history_resnet, num_epochs, train_size, fold, name_file_rede_resnet, title_graph_rede_resnet, version)
                    graphs.loss_graph(history_resnet, num_epochs, train_size, fold, name_file_rede_resnet, title_graph_rede_resnet, version)
                    auc_all_rn, fpr_all_rn, tpr_all_rn, lauc_rn = graphs.roc_graph('resnet', model_resnet, x_test, y_test, title_graph_rede_resnet, train_size, fold, version, name_file_rede_resnet, auc_all_rn, fpr_all_rn, tpr_all_rn, lauc_rn)
                    prec_all_rn, rec_all_rn, f_1_score_all_rn, f_100_score_all_rn = graphs.fscore_graph('resnet', model_resnet, x_test, y_test, title_graph_rede_resnet, train_size, fold, version, name_file_rede_resnet, prec_all_rn, rec_all_rn, f_1_score_all_rn, f_100_score_all_rn)  #(y_test, x_test, model_resnet, 'resnet')

                    print('\n ** Training EfficientNet.')
                    callbacks_efn = compile_model_effnet(name_file_rede, model_effnet, opt_efn, fold, version)
                    history_efn = fit_model_effnet(model_effnet, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks_efn)
                    print('\n ** Training EfficientNet completed.')
                    print(' ** Plotting Efficient Net Graphs')
                    graphs.accurary_graph(history_efn, num_epochs, train_size, fold, name_file_rede_effnet, title_graph_rede_effnet, version)
                    graphs.loss_graph(history_efn, num_epochs, train_size, fold, name_file_rede_effnet, title_graph_rede_effnet, version)
                    auc_all_ef, fpr_all_ef, tpr_all_ef, lauc_ef = graphs.roc_graph('effnet', model_effnet, x_test, y_test, title_graph_rede_effnet, train_size, fold, version, name_file_rede_effnet, auc_all_ef, fpr_all_ef, tpr_all_ef, lauc_ef)
                    prec_all_ef, rec_all_ef, f_1_score_all_ef, f_100_score_all_ef = graphs.fscore_graph('effnet', model_effnet, x_test, y_test, title_graph_rede_effnet, train_size, fold, version, name_file_rede_effnet, prec_all_ef, rec_all_ef, f_1_score_all_ef, f_100_score_all_ef)

                    if dirichlet:

                        model_1 = KerasMember(name='ResNet', keras_model=model_resnet, train_batches=(x_data_cv, y_data_cv), val_batches=(x_val_cv, y_val_cv))
                        model_2 = KerasMember(name='EffNet', keras_model=model_effnet, train_batches=(x_data_cv, y_data_cv), val_batches=(x_val_cv, y_val_cv))

                        wAvgEnsemble = DirichletEnsemble()
                        wAvgEnsemble.add_members([model_1, model_2])
                        #wAvgEnsemble.fit()
                        Dir_AUC = wAvgEnsemble.bestscore
                        print(" ** Dir_AUC: ", Dir_AUC)
                        wAvgEnsemble.fit() #_generator(generator, steps_per_epoch=len(x_data_cv) / batch_size,
                        #epochs=num_epochs, verbose=1, validation_data=(x_val_cv, y_val_cv),
                        #validation_steps=len(x_val_cv) / batch_size, callbacks=callbacks)
                        print('\n ** Using Dirichlet Ensemble:')
                        dir_AUC_all = np.append(dir_AUC_all, wAvgEnsemble.bestscore)
                        Dir_AUC = wAvgEnsemble.bestscore
                        print(" ** Dir_AUC: ", Dir_AUC)
                        wAvgEnsemble.describe()
                        print(wAvgEnsemble)

                    if logistic:
                        log_AUC = stacked_logisticregression_ensemble(model_resnet, model_effnet, x_test, y_test[:, 1])
                        log_AUC_all = np.append(log_AUC_all, log_AUC)

                    if integrated_stacked:
                        integ_AUC = integrated_stacked_model(model_resnet, model_effnet, x_test, y_test, optimizer)
                        print("Integrated Stacked Models AUC", integ_AUC)
                        integs_AUC_all = np.append(integs_AUC_all, integ_AUC)

                    # Generate ROC Curve
                    tpr, fpr, auc, auc2, thres = utils.roc_curve_calculate_ensemble(y_test, x_test, model_resnet, model_effnet, rede, version)
                    lauc = np.append(lauc, auc)
                    auc_all.append(auc2)
                    fpr_all.append(fpr)
                    tpr_all.append(tpr)
                    print('\n ** Plotting Ensemble ROC Graph')
                    graphs.roc_graph(fpr_all, tpr_all, auc, train_size, fold, name_file_rede, title_graph_rede, version)
                    auc_all, fpr_all, tpr_all, lauc, thres = graphs.roc_graph_ensemble(rede, model_resnet, model_effnet, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede, auc_all, fpr_all, tpr_all, lauc)
                    prec_all, rec_all, f_1_score_all, f_100_score_all = graphs.fscore_graph_ensemble(rede, model_resnet, model_effnet, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede, prec_all, rec_all, f_1_score_all, f_100_score_all)

                    # Resultados Efficient Net

                    # TODO - VERIFICAR ESSE TRECHO...NÃO FAZ SENTIDO!!! - parece não ter muita influência
                    # Resnet
                    acc0_res, val_acc0_res, loss0_res, val_loss0_res = utils.acc_score(acc0_res, history_resnet, val_acc0_res, loss0_res, val_loss0_res)

                    # Efficient Net
                    acc0_eff, val_acc0_eff, loss0_eff, val_loss0_eff = utils.acc_score(acc0_eff, history_efn, val_acc0_eff, loss0_eff, val_loss0_eff)

                    scores_resnet = model_resnet.evaluate(x_test, y_test, verbose=0)
                    print(' ** RESNET - Large CNN Error: %.2f%%' % (100 - scores_resnet[1] * 100))
                    scores_efn = model_effnet.evaluate(x_test, y_test, verbose=0)
                    print(' ** Efficient Net - Large CNN Error: %.2f%%' % (100 - scores_efn[1] * 100))

                elaps = (time.perf_counter() - foldtimer) / 60
                print('\n ** Fold TIME: %.3f minutes.' % elaps)

        # CLOSE Loop para a execução de todas as FOLDS
        print('\n ** Training and evaluation complete.')
        elapsed = (time.perf_counter() - start) / 60
        print(' ** %.3f TIME: %.3f minutes.' % (train_size, elapsed))

        if rede == 'ensemble':
            highauc_rn, lowauc_rn, mauc_rn = graphs.ultimate_ROC(lauc_rn, auc_all_rn, thres, tpr_all_rn, fpr_all_rn, name_file_rede_resnet, title_graph_rede_resnet, k_folds, train_size, 'resnet', version)
            highauc_ef, lowauc_ef, mauc_ef = graphs.ultimate_ROC(lauc_ef, auc_all_ef, thres, tpr_all_ef, fpr_all_ef, name_file_rede_effnet, title_graph_rede_effnet, k_folds, train_size, 'effnet', version)
        
        highauc, lowauc, mauc = graphs.ultimate_ROC(lauc, auc_all, thres, tpr_all, fpr_all, name_file_rede, title_graph_rede, k_folds, train_size, rede, version)

        if rede == 'ensemble':
            if dirichlet:
                med_dir_AUC = (np.percentile(dir_AUC_all, 50.0))
                low_dir_AUC = (np.percentile(dir_AUC_all, 15.87))
                hig_dir_AUC = (np.percentile(dir_AUC_all, 84.13))
            else:
                med_dir_AUC, low_dir_AUC, hig_dir_AUC = [0 for i in range(3)]
            if logistic:
                med_log_AUC = (np.percentile(log_AUC_all, 50.0))
                low_log_AUC = (np.percentile(log_AUC_all, 15.87))
                hig_log_AUC = (np.percentile(log_AUC_all, 84.13))
            else:
                med_log_AUC, low_log_AUC, hig_log_AUC = [0 for i in range(3)]
            if integrated_stacked:
                med_int_AUC = (np.percentile(integs_AUC_all, 50.0))
                low_int_AUC = (np.percentile(integs_AUC_all, 15.87))
                hig_int_AUC = (np.percentile(integs_AUC_all, 84.13))
            else:
                med_int_AUC, low_int_AUC, hig_int_AUC = [0 for i in range(3)]
            code_data = [train_size, lowauc_rn, mauc_rn, highauc_rn,
                lowauc_ef, mauc_ef, highauc_ef, low_dir_AUC, med_dir_AUC, hig_dir_AUC,
                low_log_AUC, med_log_AUC, hig_log_AUC, low_int_AUC, med_int_AUC, hig_int_AUC, 
                lowauc, mauc, highauc, prec_all, rec_all, f_1_score_all, f_100_score_all,
                prec_all_rn, rec_all_rn, f_1_score_all_rn, f_100_score_all_rn,
                prec_all_ef, rec_all_rn, f_1_score_all_ef, f_100_score_all_ef]
            print(" THIS IS CODE_DATA!")
            print(code_data)
            writer.writerow(code_data)

        # Movendo os arquivos para as pastas desejadas - TODO PARECE OK NO ENSEMBLE!!
        print(' ** ISSO FUNCIONA???')
        if rede == 'ensemble':
            files_changer.filemover(train_size, version, k_folds, name_file_rede_resnet, title_graph_rede_resnet, num_epochs)
            files_changer.filemover(train_size, version, k_folds, name_file_rede_effnet, title_graph_rede_effnet, num_epochs)
            files_changer.filemover(train_size, version, k_folds, name_file_rede, title_graph_rede, num_epochs)
        else:
            files_changer.filemover(train_size, version, k_folds, name_file_rede, title_graph_rede, num_epochs)

        k_back.clear_session()

        hlow_auc = np.append(hlow_auc, lowauc)
        hhigh_auc = np.append(hhigh_auc, highauc)
        hmed_auc = np.append(hmed_auc, mauc)

        time_fold = (time.perf_counter() - begin_fold) / (60 * 60)
        print('\n Ciclo ', u, ' concluido em: ', time_fold)

    #except AssertionError as error:
    #    print(error)


files_changer.last_mover(version)

time_total = (time.perf_counter() - begin) / (60 * 60)
print('\n ** Mission accomplished in {} hours.'. format(time_total))
print('\n ** FINISHED! ************************')
