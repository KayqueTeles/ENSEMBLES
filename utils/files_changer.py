import os
import shutil
import tarfile
import zipfile
import wget

path_folder = '/home/kayque/LENSLOAD/RESULTS'
# path_folder = 'C:/Users/Teletrabalho/Documents/Pessoal/resultados'
# path_folder = '/home/hdd2T/icaroDados'


# Método que faz o download do dataset, se necessário
def data_downloader():
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

def file_cleaner(k_folds, version, name_file_rede, num_epochs, challenge_size):
    piccounter, weicounter, csvcounter = (0 for i in range(3))
    print('\n ** Removing specified files and folders...')
    print(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    for fold in range(0, k_folds, 1):
        for train_size in range(challenge_size):
            # fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size))
            if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version)):
                os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version))
                piccounter = piccounter + 1
            # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}.png'. format(name_file_rede, train_size, fold))
            if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('AccxEpoch_{}_{}_Fold_{}.png'. format(name_file_rede, train_size, fold))
            if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
            if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
            if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
            if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
                os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version))
                piccounter = piccounter + 1
            if os.path.exists('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            if os.path.exists('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
            if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
                os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version))
                piccounter = piccounter + 1
            if os.path.exists('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            if os.path.exists('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                os.remove('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                piccounter = piccounter + 1
            # plt.savefig('AUCxSize_{}_version_{}.png'. format(name_file_rede))
            if os.path.exists('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version)):
                os.remove('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version))
                piccounter = piccounter + 1
            # csv_name = 'training_{}_fold_{}.csv'. format(name_file_rede, fold) -- Resnet e Effnet
            if os.path.exists('training_{}_fold_{}.csv'. format(name_file_rede, fold)):
                os.remove('training_{}_fold_{}.csv'. format(name_file_rede, fold))
                csvcounter = csvcounter + 1
            # csv_name_resnet = 'training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)
            if os.path.exists('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)):
                os.remove('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold))
                csvcounter = csvcounter + 1
            # csv_na   me_efn = 'training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)
            if os.path.exists('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)):
                os.remove('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold))
                csvcounter = csvcounter + 1
            if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold)):
                os.remove('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold))
                piccounter = piccounter + 1  

    # for (epoch = 0  epoch):
    for epoch in range(0, num_epochs, 1):
        # Números menores que 10
        if os.path.exists('Train_model_weights_{}_0{}_{}.h5'. format(name_file_rede, epoch + 1, version)):
            os.remove('Train_model_weights_{}_0{}_{}.h5'. format(name_file_rede, epoch + 1, version))
            weicounter = weicounter + 1
        # Números maiores que 10
        elif os.path.exists('Train_model_weights_{}_{}_{}.h5'. format(name_file_rede, epoch + 1, version)):
            os.remove('Train_model_weights_{}_{}_{}.h5'. format(name_file_rede, epoch +1, version))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_weights_{}_0{}.h5'. format(name_file_rede, epoch + 1)):
            os.remove('Train_model_weights_{}_0{}.h5'. format(name_file_rede, epoch + 1))
            weicounter = weicounter + 1
        # Números maiores que 10
        elif os.path.exists('Train_model_weights_{}_{}.h5'. format(name_file_rede, epoch + 1)):
            os.remove('Train_model_weights_{}_{}.h5'. format(name_file_rede, epoch +1))
            weicounter = weicounter + 1

    # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
    if os.path.exists('{}_{}_{}/'. format(name_file_rede, version, train_size)):
        shutil.rmtree('{}_{}_{}/'. format(name_file_rede, version, train_size))

    print('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(piccounter, weicounter, csvcounter))


# Método que elimina arquivos de testes anteriores
def file_remover(train_size, k_folds, version, name_file_rede, num_epochs):
    piccounter, weicounter, csvcounter = (0 for i in range(3))
    print('\n ** Removing specified files and folders...')
    print(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    for fold in range(0, k_folds, 1):
        # fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size))
        if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version)):
            os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version))
            piccounter = piccounter + 1
        # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        # plt.savefig('AccxEpoch_{}_{}_Fold_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
        if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
            os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version))
            piccounter = piccounter + 1
        if os.path.exists('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        if os.path.exists('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
        if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
            os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version))
            piccounter = piccounter + 1
        if os.path.exists('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        if os.path.exists('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            os.remove('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
            piccounter = piccounter + 1
        # plt.savefig('AUCxSize_{}_version_{}.png'. format(name_file_rede))
        if os.path.exists('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version)):
            os.remove('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version))
            piccounter = piccounter + 1
        # csv_name = 'training_{}_fold_{}.csv'. format(name_file_rede, fold) -- Resnet e Effnet
        if os.path.exists('training_{}_fold_{}.csv'. format(name_file_rede, fold)):
            os.remove('training_{}_fold_{}.csv'. format(name_file_rede, fold))
            csvcounter = csvcounter + 1
        # csv_name_resnet = 'training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)
        if os.path.exists('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)):
            os.remove('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold))
            csvcounter = csvcounter + 1
        # csv_name_efn = 'training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)
        if os.path.exists('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)):
            os.remove('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold))
            csvcounter = csvcounter + 1

    # for (epoch = 0  epoch):
    for epoch in range(0, num_epochs, 1):
        # Números menores que 10
        if os.path.exists('Train_model_weights_{}_0{}_{}.h5'. format(name_file_rede, epoch + 1, version)):
            os.remove('Train_model_weights_{}_0{}_{}.h5'. format(name_file_rede, epoch + 1, version))
            weicounter = weicounter + 1
        # Números maiores que 10
        elif os.path.exists('Train_model_weights_{}_{}_{}.h5'. format(name_file_rede, epoch + 1, version)):
            os.remove('Train_model_weights_{}_{}_{}.h5'. format(name_file_rede, epoch +1, version))
            weicounter = weicounter + 1
        if os.path.exists('Train_model_weights_{}_0{}.h5'. format(name_file_rede, epoch + 1)):
            os.remove('Train_model_weights_{}_0{}.h5'. format(name_file_rede, epoch + 1))
            weicounter = weicounter + 1
        # Números maiores que 10
        elif os.path.exists('Train_model_weights_{}_{}.h5'. format(name_file_rede, epoch + 1)):
            os.remove('Train_model_weights_{}_{}.h5'. format(name_file_rede, epoch +1))
            weicounter = weicounter + 1

    # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
    if os.path.exists('{}_{}_{}/'. format(name_file_rede, version, train_size)):
        shutil.rmtree('{}_{}_{}/'. format(name_file_rede, version, train_size))

    print('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(piccounter, weicounter, csvcounter))


def filemover(train_size, version, k_folds, name_file_rede, title_graph_rede, num_epochs):
    # Checando se as pastas já existem, e apagando ou criando, de acordo com o caso
    print('\n ** Moving created files to a certain folder.')
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

    # print(" ** Checking if there's an RNCV folder...")
    print(" ** Checking if there's an network({}) folder...". format(title_graph_rede))
    # if os.path.exists('/home/dados2T/icaroDados/resnetLens/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, train_size)):
    if os.path.exists('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, name_file_rede, version, train_size)):
        # if os.path.exists('results/RNCV_%s_%s' % (version, train_size)):
        print(' ** Yes. There is. Trying to delete and renew...')
        shutil.rmtree('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, name_file_rede, version, train_size))
        # shutil.rmtree('results/RNCV_%s_%s' % (version, TR))
        os.mkdir('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, name_file_rede, version, train_size))
        # os.mkdir('results/RNCV_%s_%s' % (version, TR))
        print(' ** Done!')
    else:
        print(' ** None found. Creating one.')
        # os.mkdir('results')
        os.mkdir('./RESULTS/ENSEMBLE_{}/{}_{}_{}/'. format(version, name_file_rede, version, train_size))
        # os.mkdir('results/RNCV_%s_%s' % (version, TR))
        print(' ** Done!')

        # dest1 = ('/home/kayque/LENSLOAD/RESULTS/ENSEMBLE_{}/RNCV_%s_%s/' % (version, TR))
    dest1 = ('{}/ENSEMBLE_{}/{}_{}_{}/'. format(path_folder, version, name_file_rede, version, train_size))

    # Variável contador de weights removidos
    weicounter = 0

    # Movendo os arquivos criados na execução do teste
    for fold in range(0, 10 * k_folds, 1):
        # fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size))
        if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version)):
            shutil.move('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version), dest1)
            counter = counter + 1
        # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        # plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
        if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
        if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
            shutil.move('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version), dest1)
            counter = counter + 1
        # plt.savefig('AUCxSize_{}_version_{}.png'. format(name_file_rede))
        if os.path.exists('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version)):
            shutil.move('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version), dest1)
            counter = counter + 1
        # csv_name = 'training_{}_fold_{}.csv'. format(name_file_rede, fold) -- Resnet e Effnet
        if os.path.exists('training_{}_fold_{}.csv'. format(name_file_rede, fold)):
            shutil.move('training_{}_fold_{}.csv'. format(name_file_rede, fold), dest1)
            counter = counter + 1
        # csv_name_resnet = 'training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)
        if os.path.exists('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)):
            shutil.move('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold), dest1)
            counter = counter + 1
        # csv_name_efn = 'training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)
        if os.path.exists('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)):
            shutil.move('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold), dest1)
            counter = counter + 1
        if os.path.exists('ensemble_data_{}.csv'. format(version)):
            shutil.move('ensemble_data_{}.csv'. format(version), dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        if os.path.exists('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
        if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
            shutil.move('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version), dest1)
            counter = counter + 1
        if os.path.exists('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        if os.path.exists('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
            shutil.move('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version), dest1)
            counter = counter + 1
        if os.path.exists('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold)):
            shutil.move('./CLUE_FROM_DATASET_{}_samples_{}_version_{}_step_{}x{}_size_{}_num.png'. format(train_size, version, 'generator', 66, 66, fold))
            counter = counter + 1  
        for epoch in range(0, num_epochs, 1):
            epo = epoch + 1
            if epoch < 10:
                # Números menores que 10
                if os.path.exists('./Train_model_weights_{}_0{}_ver_{}.h5'. format(name_file_rede, epo, version)):
                    os.remove('./Train_model_weights_{}_0{}_ver_{}.h5'. format(name_file_rede, epo, version))
                    weicounter = weicounter + 1
            else:
                # Números maiores que 10
                if os.path.exists('./Train_model_weights_{}_{}_ver_{}.h5'. format(name_file_rede, epo, version)):
                    os.remove('./Train_model_weights_{}_{}_ver_{}.h5'. format(name_file_rede, epo, version))
                    weicounter = weicounter + 1

    print('\n ** Done. {} files moved, '. format(counter), '{} weights removed.'. format(weicounter))

def last_mover(version):
    print('\n ** Process is almost finished.')
    print(' ** Proceeding to move LAST files to RESULTS folder.')
    counter = 0
    print(" ** Checking if there's a RESULTS folder...")

    dest3 = ('/home/kayque/LENSLOAD/RESULTS/ENSEMBLE_%s' % version)
    print(' ** Initiating last_mover...')
    if os.path.exists('code_data_version_{}.csv'. format(version)):
        shutil.move('code_data_version_{}.csv'. format(version), dest3)
        counter = counter + 1
    if os.path.exists('code_parameters_version_{}.csv'. format(version)):
        shutil.move('code_parameters_version_{}.csv'. format(version), dest3)
        counter = counter + 1

    print(" ** Moving done. %s files moved." % counter)

def massive_cleaner(k_folds, name_file_rede, num_epochs, challenge_size):
    piccounter, weicounter, csvcounter = (0 for i in range(3))
    print('\n ** Removing specified files and folders...')
    print(' ** Checking .png, .h5 and csv files...')
    # for fold in range(0, 10 * k_folds, 1):
    for version in range(300):
        for fold in range(0, k_folds, 1):
            for train_size in range(challenge_size):
                # fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size))
                if os.path.exists('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version)):
                    os.remove('./TrainTest_rate_{}_TR_{}_version_{}.png'. format(name_file_rede, train_size, version))
                    piccounter = piccounter + 1
                # fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}.png'. format(name_file_rede, train_size, fold))
                if os.path.exists('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                # plt.savefig('AccxEpoch_{}_{}_Fold_{}.png'. format(name_file_rede, train_size, fold))
                if os.path.exists('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./AccxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                # plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
                if os.path.exists('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./LossxEpoch_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                # plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold))
                if os.path.exists('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                # plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
                if os.path.exists('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
                    os.remove('./ROCLensDetectNet_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version))
                    piccounter = piccounter + 1
                if os.path.exists('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                if os.path.exists('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./ROCLensDetectNet_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                # plt.savefig('AUCxSize_{}_version_{}.png'. format(name_file_rede))
                if os.path.exists('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version)):
                    os.remove('./AUCxSize_{}_version_{}.png'. format(name_file_rede, version))
                    piccounter = piccounter + 1
                # csv_name = 'training_{}_fold_{}.csv'. format(name_file_rede, fold) -- Resnet e Effnet
                if os.path.exists('training_{}_fold_{}.csv'. format(name_file_rede, fold)):
                    os.remove('training_{}_fold_{}.csv'. format(name_file_rede, fold))
                    csvcounter = csvcounter + 1
                # csv_name_resnet = 'training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)
                if os.path.exists('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold)):
                    os.remove('training_{}_resnet_fold_{}.csv'. format(name_file_rede, fold))
                    csvcounter = csvcounter + 1
                # csv_na   me_efn = 'training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)
                if os.path.exists('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold)):
                    os.remove('training_{}_effnet_fold_{}.csv'. format(name_file_rede, fold))
                    csvcounter = csvcounter + 1
                if os.path.exists('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                # plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size))
                if os.path.exists('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version)):
                    os.remove('./FScoreGraph_{}_Full_{}_version_{}.png'. format(name_file_rede, train_size, version))
                    piccounter = piccounter + 1
                if os.path.exists('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./FScoreGraph_{}_resnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1
                if os.path.exists('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version)):
                    os.remove('./FScoreGraph_{}_effnet_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
                    piccounter = piccounter + 1

        # for (epoch = 0  epoch):
        for epoch in range(0, num_epochs, 1):
            # Números menores que 10
            if os.path.exists('Train_model_weights_{}_0{}_{}.h5'. format(name_file_rede, epoch + 1, version)):
                os.remove('Train_model_weights_{}_0{}_{}.h5'. format(name_file_rede, epoch + 1, version))
                weicounter = weicounter + 1
            # Números maiores que 10
            elif os.path.exists('Train_model_weights_{}_{}_{}.h5'. format(name_file_rede, epoch + 1, version)):
                os.remove('Train_model_weights_{}_{}_{}.h5'. format(name_file_rede, epoch +1, version))
                weicounter = weicounter + 1
            if os.path.exists('Train_model_weights_{}_0{}.h5'. format(name_file_rede, epoch + 1)):
                os.remove('Train_model_weights_{}_0{}.h5'. format(name_file_rede, epoch + 1))
                weicounter = weicounter + 1
            # Números maiores que 10
            elif os.path.exists('Train_model_weights_{}_{}.h5'. format(name_file_rede, epoch + 1)):
                os.remove('Train_model_weights_{}_{}.h5'. format(name_file_rede, epoch +1))
                weicounter = weicounter + 1

    print('\n ** Done. {} .png files removed, {} .h5 files removed and {} .csv removed.'. format(piccounter, weicounter, csvcounter))