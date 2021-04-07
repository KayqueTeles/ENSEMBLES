import os
import shutil
import tarfile
import zipfile
import wget
import numpy as np
import time
import h5py, pandas as pd
import itertools
from utils import utils
from joblib import Parallel, delayed
import os

path_folder = '/home/kayque/LENSLOAD/RESULTS'
# path_folder = 'C:/Users/Teletrabalho/Documents/Pessoal/resultados'
# path_folder = '/home/hdd2T/icaroDados'

# Método que faz o download do dataset, se necessário
def data_downloader(num_samples, version, input_shape):
    foldtimer = time.perf_counter()

    PATH = os.getcwd()
    var = PATH + '/' + 'lensdata/'

    if os.path.exists('./lensdata/x_data20000fits.h5'):
        print(' ** Files from lensdata.tar.gz were already downloaded.')
    else:
        print('\n ** Downloading lensdata.zip...')
        wget.download('https://clearskiesrbest.files.wordpress.com/2019/02/lensdata.zip')
        print(' ** Download successful. Extracting...')
        with zipfile.ZipFile('lensdata.zip', 'r') as zip_ref:
            zip_ref.extractall()
            print(' ** Extracted successfully.')
        print(' ** Extracting data from lensdata.tar.gz...')
        tar = tarfile.open('lensdata.tar.gz', 'r:gz')
        tar.extractall()
        tar.close()
        print(' ** Extracted successfully.')

    if os.path.exists('./lensdata/x_data20000fits.h5'):
        print(' ** Files from lensdata.tar.gz were already extracted.')
    else:
        print(' ** Extracting data from #DataVisualization.tar.gz...')
        tar = tarfile.open('./lensdata/DataVisualization.tar.gz', 'r:gz')
        tar.extractall('./lensdata/')
        tar.close()
        print(' ** Extracted successfully.')
        print(' ** Extrating data from x_data20000fits.h5.tar.gz...')
        tar = tarfile.open('./lensdata/x_data20000fits.h5.tar.gz', 'r:gz')
        tar.extractall('./lensdata/')
        tar.close()
        print(' ** Extracted successfully.')
    if os.path.exists('lensdata.tar.gz'):
        os.remove('lensdata.tar.gz')
    if os.path.exists('lensdata.zip'):
        os.remove('lensdata.zip')

    for pa in range(0, 10, 1):
        if os.path.exists('lensdata ({}).zip'. format(pa)):
            os.remove('lensdata ({}).zip'. format(pa))
    print('\n ** Starting data preprocessing...')
    labels = pd.read_csv(var + 'y_data20000fits.csv', delimiter=',', header=None)
    y_data = np.array(labels, np.uint8)
    y_data = y_data[:num_samples]
    print(y_data)

    x_datasaved = h5py.File(var + 'x_data20000fits.h5', 'r')
    Ni_channels = 0  # first channel
    N_channels = 3  # number of channels
    channels = ['R', 'G', 'U']

    x_data = x_datasaved['data']
    x_data = x_data[:, :, :, Ni_channels:Ni_channels + N_channels]
    x_data = x_data[:num_samples, :, :, :]

    index = utils.save_clue(x_data, y_data, num_samples, version, 'generator', input_shape, 5, 5, 0, channels)

    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Data Generation TIME: %.3f minutes.' % elaps)

    return (x_data, y_data, index, channels)

def file_cleaner(k_folds, version, input_shape, num_epochs, challenge_size):
    counter, weicounter, csvcounter = (0 for i in range(3))
    foldtimer = time.perf_counter()
    print('\n ** Removing specified files and folders...')
    print(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    #models_list = np.append(model_list, 'ensemble')
    models_list = ['resnet50', 'resnet101', 'resnet152', 'ensemble',
                   'effnet_B0', 'effnet_B1', 'effnet_B2', 'effnet_B3',
                   'effnet_B4', 'effnet_B5', 'effnet_B6', 'effnet_B7',
                   'inceptionV2', 'inceptionV3', 'xception']
    challist = np.linspace(0, challenge_size, (challenge_size/10))
    for mod in models_list:
        print(" -- Model: ", mod)
        for train_size in challist:
            for fold in range(int(k_folds)+1):
                if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
                if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
                if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                    os.remove('./AUCxSize_{}_version_{}.png'. format(mod, version))
                    counter = counter + 1
                # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
                if os.path.exists('./training_{}_fold_{}.csv'. format(mod, fold)):
                    os.remove('./training_{}_fold_{}.csv'. format(mod, fold))
                    counter = counter + 1
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                    os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                    counter = counter + 1
                # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
                if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                    os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                    counter = counter + 1
                if os.path.exists('./code_data_version_{}_model_{}_aucs.csv'. format(version, mod)):
                    os.remove('./code_data_version_{}_model_{}_aucs.csv'. format(version, mod))
                    csvcounter = csvcounter + 1
                if os.path.exists('./code_data_version_{}.csv'. format(version)):
                    os.remove('./code_data_version_{}.csv'. format(version))
                    csvcounter = csvcounter + 1
                if os.path.exists('./code_parameters_version_{}.csv'. format(version)):
                    os.remove('./code_parameters_version_{}.csv'. format(version))
                    csvcounter = csvcounter + 1
                for jk in range(100*int(k_folds)):
                    if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(challenge_size, version, 'generator', input_shape, input_shape, jk)):
                        os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(challenge_size, version, 'generator', input_shape, input_shape, jk))
                        counter = counter + 1 
                for epoch in range(int(num_epochs)):
                    epo = epoch + 1
                    if epoch < 10:
                    # Números menores que 10
                        if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version)):
                            os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version))
                            weicounter = weicounter + 1
                    else:
                        # Números maiores que 10
                        if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version)):
                            os.remove('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version))
                            weicounter = weicounter + 1

            # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
            if os.path.exists('{}_{}_{}/'. format(mod, version, train_size)):
                shutil.rmtree('{}_{}_{}/'. format(mod, version, train_size))

    print('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(counter, weicounter, csvcounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Removing TIME: %.3f minutes.' % elaps)


# Método que elimina arquivos de testes anteriores
def file_remover(train_size, k_folds, version, model_list, num_epochs):
    piccounter, weicounter, csvcounter = (0 for i in range(3))
    foldtimer = time.perf_counter()
    print(' ** Itiating file_remover for removal of specified files and folders...')
    print(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    models_list = np.append(model_list, 'ensemble')
    for mod in models_list:
        for fold in range(int(k_folds)+1):
            if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version))
                piccounter = piccounter + 1
            # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
            if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                piccounter = piccounter + 1
            # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
            if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                os.remove('./AUCxSize_{}_version_{}.png'. format(mod, version))
                piccounter = piccounter + 1
            # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
            if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                os.remove('training_{}_fold_{}.csv'. format(mod, fold))
                piccounter = piccounter + 1
            if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
            if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                piccounter = piccounter + 1
            for epoch in range(int(num_epochs)):
                epo = epoch + 1
                if epoch < 10:
                # Números menores que 10
                    if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version)):
                        os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version))
                        weicounter = weicounter + 1
                else:
                    # Números maiores que 10
                    if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version)):
                        os.remove('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version))
                        weicounter = weicounter + 1

        # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
        if os.path.exists('{}_{}_{}/'. format(mod, version, train_size)):
            shutil.rmtree('{}_{}_{}/'. format(mod, version, train_size))

    print('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(piccounter, weicounter, csvcounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Removing TIME: %.3f minutes.' % elaps)


def filemover(train_size, version, k_folds, model_list, num_epochs):
    # Checando se as pastas já existem, e apagando ou criando, de acordo com o caso
    print('\n ** Moving created files to a certain folder.')
    foldtimer = time.perf_counter()
    counter = 0

    print(" ** Checking if there's a results folder...")
    if os.path.exists('./RESULTS'):
        print(' ** results file found. Moving forward.')
    else:
        print(' ** None found. Creating one.')
        os.mkdir('./RESULTS')
        print(' ** Done!')
    print(" ** Checking if there's a local folder...")
    if os.path.exists('RESULTS/ENSEMBLE_%s' % version):
        print(' ** Yes. There is.')
    else:
        print(" ** None found. Creating one.")
        os.mkdir('RESULTS/ENSEMBLE_%s' % version)
        print(" ** Done!")
    weicounter = 0
    models_list = np.append(model_list, 'ensemble')
    print(models_list)

    for mod in models_list:
        print(" ** Checking if there's an network({}) folder...". format(mod))
        if os.path.exists('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size)):
            # if os.path.exists('results/RNCV_%s_%s' % (version, train_size)):
            print(' ** Yes. There is. Trying to delete and renew...')
            shutil.rmtree('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size))
            os.mkdir('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size))
            # os.mkdir('results/RNCV_%s_%s' % (version, TR))
            print(' ** Done!')
        else:
            print(' ** None found. Creating one.')
            os.mkdir('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, mod, version, train_size))
            print(' ** Done!')
        
        dest1 = ('{}/ENSEMBLE_{}/{}_{}_{}/'. format(path_folder, version, mod, version, train_size))

        # Movendo os arquivos criados na execução do teste
        for fold in range(int(k_folds)+1):
            # fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size))
            if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                shutil.move('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version), dest1)
                counter = counter + 1
            # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
            if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
            if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                shutil.move('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version), dest1)
                counter = counter + 1
            # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
            if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                shutil.move('./AUCxSize_{}_version_{}.png'. format(mod, version), dest1)
                counter = counter + 1
            # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
            if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                shutil.move('training_{}_fold_{}.csv'. format(mod, fold), dest1)
                counter = counter + 1
            if os.path.exists('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            if os.path.exists('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                shutil.move('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version), dest1)
                counter = counter + 1
            # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
            if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                shutil.move('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version), dest1)
                counter = counter + 1
            for epoch in range(int(num_epochs)):
                epo = epoch + 1
                #if epoch < 10:
                # Números menores que 10
                    #if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version)):
                    #    os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version))
                    #    weicounter = weicounter + 1
                #else:
                    # Números maiores que 10
                    #if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version)):
                    #    os.remove('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version))
                    #    weicounter = weicounter + 1

    print('\n ** Done. {} files moved, '. format(counter), '{} weights removed.'. format(weicounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Moving TIME: %.3f minutes.' % elaps)

def last_mover(version, model_list, dataset_size, num_epochs, input_shape):
    print('\n ** Process is almost finished.')
    print(' ** Proceeding to move LAST files to RESULTS folder.')
    counter = 0
    print(" ** Checking if there's a RESULTS folder...")

    dest3 = ('/home/kayque/LENSLOAD/RESULTS/ENSEMBLE_%s' % version)
    print(' ** Initiating last_mover...')
    models_list = np.append(model_list, 'ensemble')
    for mo in models_list:
        if os.path.exists('code_data_version_{}_model_{}_aucs.csv'. format(version, mo)):
            shutil.move('code_data_version_{}_model_{}_aucs.csv'. format(version, mo), dest3)
            counter = counter + 1
        if os.path.exists('code_data_version_{}_model_{}_f1s.csv'. format(version, mo)):
            shutil.move('code_data_version_{}_model_{}_f1s.csv'. format(version, mo), dest3)
            counter = counter + 1
        for g in range(dataset_size):
            if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(dataset_size, version, 'generator', input_shape, input_shape, int(g))):
                shutil.move('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(dataset_size, version, 'generator', input_shape, input_shape, int(g)), dest3)
                counter = counter + 1
        for epoch in range(int(num_epochs)):
            epo = epoch + 1
            #if epoch < 10:
            # Números menores que 10
                #if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(mo, epo, version)):
                #    os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(mo, epo, version))
            #else:
                # Números maiores que 10
                #if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(mo, epo, version)):
                #    os.remove('./Train_model_weights_{}_{}_{}.h5'. format(mo, epo, version))

    if os.path.exists('code_parameters_version_{}.csv'. format(version)):
        shutil.move('code_parameters_version_{}.csv'. format(version), dest3)
        counter = counter + 1

    print(" ** Moving done. %s files moved." % counter)

def massive_cleaner(k_folds, num_epochs, challenge_size):
    counter, weicounter, csvcounter = (0 for i in range(3))
    foldtimer = time.perf_counter()
    print('\n ** Removing specified files and folders...')
    print(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    models_list = ['resnet50', 'resnet101', 'resnet152', 'ensemble',
                   'effnet_B0', 'effnet_B1', 'effnet_B2', 'effnet_B3',
                   'effnet_B4', 'effnet_B5', 'effnet_B6', 'effnet_B7',
                   'inceptionV2', 'inceptionV3', 'xception']
    for mod in models_list:
        print(" -- Model: ", mod)
        for version in range(300):
            for train_size in range(challenge_size):
                for fold in range(int(k_folds)+1):
                    if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                        os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version))
                        counter = counter + 1
                    # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
                    if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                        os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                        counter = counter + 1
                    # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
                    if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                        os.remove('./AUCxSize_{}_version_{}.png'. format(mod, version))
                        counter = counter + 1
                    # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
                    if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                        os.remove('training_{}_fold_{}.csv'. format(mod, fold))
                        counter = counter + 1
                    if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
                    if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                        os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                        counter = counter + 1
                    if os.path.exists('./code_data_version_{}_model_{}_aucs.csv'. format(version, mod)):
                        os.remove('./code_data_version_{}_model_{}_aucs.csv'. format(version, mod))
                        weicounter = weicounter + 1
                    if os.path.exists('./code_data_version_{}.csv'. format(version)):
                        os.remove('./code_data_version_{}.csv'. format(version))
                        weicounter = weicounter + 1
                    if os.path.exists('./code_parameters_version_{}.csv'. format(version)):
                        os.remove('./code_parameters_version_{}.csv'. format(version))
                        weicounter = weicounter + 1
                    for jk in range(100*int(k_folds)):
                        if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(challenge_size, version, 'generator', 66, 66, jk)):
                            os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(challenge_size, version, 'generator', 66, 66, jk))
                            piccounter = piccounter + 1
                    for epoch in range(int(num_epochs)):
                        epo = epoch + 1
                        if epoch < 10:
                        # Números menores que 10
                            if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version)):
                                os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version))
                                weicounter = weicounter + 1
                        else:
                            # Números maiores que 10
                            if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version)):
                                os.remove('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version))
                                weicounter = weicounter + 1

                # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
                if os.path.exists('{}_{}_{}/'. format(mod, version, train_size)):
                    shutil.rmtree('{}_{}_{}/'. format(mod, version, train_size))

    tests = ['T', 'T2', 'T3']
    for version in tests:
        print(" -- Version: ", version)
        for mod in models_list:
            for train_size in range(challenge_size):
                for fold in range(int(k_folds)+1):
                    if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version)):
                        os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(mod, train_size, version))
                        counter = counter + 1
                    # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold))
                    if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size))
                    if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                        os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                        counter = counter + 1
                    # plt.savefig('AUCxSize_{}_version_{}.png'. format(mod))
                    if os.path.exists('./AUCxSize_{}_version_{}.png'. format(mod, version)):
                        os.remove('./AUCxSize_{}_version_{}.png'. format(mod, version))
                        counter = counter + 1
                    # csv_name = 'training_{}_fold_{}.csv'. format(mod, fold) -- Resnet e Effnet
                    if os.path.exists('training_{}_fold_{}.csv'. format(mod, fold)):
                        os.remove('training_{}_fold_{}.csv'. format(mod, fold))
                        counter = counter + 1
                    if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version)):
                        os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(mod, train_size, fold, version))
                        counter = counter + 1
                    # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size))
                    if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version)):
                        os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(mod, train_size, version))
                        counter = counter + 1
                    if os.path.exists('./code_data_version_{}_model_{}_aucs.csv'. format(version, mod)):
                        os.remove('./code_data_version_{}_model_{}_aucs.csv'. format(version, mod))
                        weicounter = weicounter + 1
                    if os.path.exists('./code_data_version_{}.csv'. format(version)):
                        os.remove('./code_data_version_{}.csv'. format(version))
                        weicounter = weicounter + 1
                    if os.path.exists('./code_parameters_version_{}.csv'. format(version)):
                        os.remove('./code_parameters_version_{}.csv'. format(version))
                        weicounter = weicounter + 1
                    for jk in range(100*int(k_folds)):
                        if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(challenge_size, version, 'generator', 66, 66, jk)):
                            os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(challenge_size, version, 'generator', 66, 66, jk))
                            piccounter = piccounter + 1
                    for epoch in range(int(num_epochs)):
                        epo = epoch + 1
                        if epoch < 10:
                        # Números menores que 10
                            if os.path.exists('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version)):
                                os.remove('./Train_model_weights_{}_0{}_{}.h5'. format(mod, epo, version))
                                weicounter = weicounter + 1
                        else:
                            # Números maiores que 10
                            if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version)):
                                os.remove('./Train_model_weights_{}_{}_{}.h5'. format(mod, epo, version))
                                weicounter = weicounter + 1

                # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
                if os.path.exists('{}_{}_{}/'. format(mod, version, train_size)):
                    shutil.rmtree('{}_{}_{}/'. format(mod, version, train_size))

    print('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(counter, weicounter, csvcounter))
    elaps = (time.perf_counter() - foldtimer) / 60
    print(' ** Removing TIME: %.3f minutes.' % elaps)