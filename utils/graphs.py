import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from utils import utils

width = 0.35
classes = ['lens', 'not-lens']


# Gráfico Completo de distribuição dos dados (DISTRIBUTION GRAPH)
def distribution_graph_complete(test_count, trainval_count, train_size, name_file_rede, title_graph_rede, version):
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(classes, test_count, width, label='Test')
    ax.bar(classes, trainval_count, width, bottom=test_count, label='Train+Val')
    ax.set_ylabel('Number of Samples')
    ax.set_title('{} - Dataset Distribution'.format(title_graph_rede))
    ax.legend(loc='lower right')
    fig.savefig('TrainTest_rate_{}_TR_{}_version_{}.png'.format(name_file_rede, train_size, version))


# Gráfico de distribuição dos dados por FOLD(DISTRIBUTION GRAPH)
def distribution_graph_fold(train_count, val_count, train_size, fold, name_file_rede, title_graph_rede, version):
    plt.figure()
    fig, ax = plt.subplots()
    ax.bar(classes, train_count, width, label='Train')
    ax.bar(classes, val_count, width, bottom=train_count, label='Validation')
    ax.set_ylabel('Number of Samples')
    ax.set_title('{} - Data distribution on fold {} with {} training samples'.format(title_graph_rede, fold, train_size))
    ax.legend(loc='lower right')
    fig.savefig('TrainVal_rate_{}_TR_{}_Fold_{}_version_{}.png'.format(name_file_rede, train_size, fold, version))


# Gráfico da Accurary
def accurary_graph(history, num_epochs, train_size, fold, name_file_rede, title_graph_rede, version):
    accu = history.history['accuracy']
    accu_val = history.history['val_accuracy']
    c = highest_integer(accu, num_epochs)

    #print(' ** accu: {}'.format(accu))
    #print(' ** accu_val: {}'.format(accu_val))

    print(' ** Plotting training & validation accuracy values.')
    plt.figure()
    plt.xlim([0, num_epochs])
    plt.ylim([0, c])
    plt.plot(accu)
    plt.plot(accu_val)
    plt.title('{} - Model Accuracy fold {}'.format(title_graph_rede, fold))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('AccxEpoch_{}_{}_Fold_{}_version_{}.png'.format(name_file_rede, train_size, fold, version))


# Gráfico da Loss
def loss_graph(history, num_epochs, train_size, fold, name_file_rede, title_graph_rede, version):
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    c = highest_integer(loss, num_epochs)

    #print('loss: {}'.format(loss))
    #print('loss_val: {}'.format(loss_val))

    print(' ** Plotting training & validation loss values.')
    plt.figure()
    plt.xlim([0, num_epochs])
    plt.ylim([0, c])
    plt.plot(loss)
    plt.plot(loss_val)
    plt.title('{} - Model Loss fold {}'.format(title_graph_rede, fold))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig('LossxEpoch_{}_{}_Fold_{}_version_{}.png'.format(name_file_rede, train_size, fold, version))

def roc_graph(rede, model, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede):
    tpr, fpr, auc, auc2, thres = utils.roc_curve_calculate(y_test, x_test, model, rede)
    print(' ** Plotting %s ROC Graph' % rede)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.plot(fpr, tpr, label='fold' + str(fold) + '& AUC: %.3f' % auc, color='C' + str(fold), linewidth=3)  # for color 'C'+str(fold), for fold[0 9]
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.title('{} - ROC with {} training samples on fold {}'.format(title_graph_rede, train_size, fold))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
    return tpr, fpr, auc, auc2, thres

def roc_graph_ensemble(rede, model_resnet, model_effnet, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede):
    tpr, fpr, auc, auc2, thres = utils.roc_curve_calculate_ensemble(y_test, x_test, model_resnet, model_effnet, rede, version)
    print(' ** Plotting %s ROC Graph' % rede)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.plot(fpr, tpr, label='fold' + str(fold) + '& AUC: %.3f' % auc, color='C' + str(fold), linewidth=3)  # for color 'C'+str(fold), for fold[0 9]
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.title('{} - ROC with {} training samples on fold {}'.format(title_graph_rede, train_size, fold))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
    return tpr, fpr, auc, auc2, thres

def roc_graphs_sec(rede, models, x_test, y_test, model_list, train_size, fold, version, name_file_rede):
    tpr, fpr, auc, auc2, thres = utils.roc_curves_sec(y_test, x_test, models, model_list, version)
    print(' ** Plotting %s ROC Graph' % rede)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.plot(fpr, tpr, label='fold' + str(fold) + '& AUC: %.3f' % auc, color='C' + str(fold), linewidth=3)  # for color 'C'+str(fold), for fold[0 9]
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.title('{} - ROC with {} training samples on fold {}'.format('ensemble', train_size, fold))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.savefig('ROCLensDetectNet_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
    return tpr, fpr, auc, auc2, thres

def final_roc_graph(k_folds, train_size, medians_x, medians_y, mauc, lowlim, highlim, name_file_rede, title_graph_rede, version):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.title('Median ROC {} over {} folds with {} training samples'.format(title_graph_rede, k_folds, train_size))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.plot(medians_x, medians_y, 'b', label='AUC: {}'.format(mauc), linewidth=3)
    plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha=0.3, interpolate=True)
    plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha=0.3, interpolate=True)
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'.format(name_file_rede, train_size, version))

def fscore_graph(rede, model, x_test, y_test, title_graph_rede, train_size, fold, version, name_file_rede):
    prec, rec, f_1_score, f_100_score, thres = utils.FScore_curves(rede, model, x_test, y_test)  #(y_test, x_test, model_resnet, 'resnet')
    print(' ** Plotting F Scores Graph')
    plt.figure()
    plt.plot([0, 1], [0.5, 0.5], 'k--')  # k = color black
    #plt.plot(rec, np.linspace(0.5, 0.5, len(rec)), 'k--')  # k = color black
    plt.plot(rec, prec, label='fold' + str(fold) + '& F1: %.3f & F100: %.3f' % (f_1_score, f_100_score), color='C' + str(fold), linewidth=3)  
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.title('{} - FScore with {} training samples on fold {}'.format(title_graph_rede, train_size, fold))
    plt.ylabel('precision', fontsize=14)
    plt.xlabel('recall', fontsize=14)
    plt.savefig('FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format(name_file_rede, train_size, fold, version))
    return prec, rec, f_1_score, f_100_score

def fscore_graph_ensemble(model_list, models, x_test, y_test, train_size, fold, version):
    prec, rec, f_1_score, f_100_score, thres = utils.FScore_curves_ensemble(y_test, x_test, models, model_list)  #(y_test, x_test, model_resnet, 'resnet')
    print(' ** Plotting F Scores Graph')
    plt.figure()
    plt.plot([0, 1], [0.5, 0.5], 'k--')  # k = color black
    #plt.plot(rec, np.linspace(0.5, 0.5, len(rec)), 'k--')  # k = color black
    plt.plot(rec, prec, label='fold' + str(fold) + '& F1: %.3f & F100: %.3f' % (f_1_score, f_100_score), color='C' + str(fold), linewidth=3)  
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.title('{} - FScore with {} training samples on fold {}'.format('ensemble', train_size, fold))
    plt.ylabel('precision', fontsize=14)
    plt.xlabel('recall', fontsize=14)
    plt.savefig('FScoreGraph_{}_{}_Fold_{}_version_{}.png'. format('ensemble', train_size, fold, version))
    return prec, rec, f_1_score, f_100_score

def plot_grafico_final(study, err, name_file_rede, title_graph_rede, version):
    lim_superior = 1.0

    plt.figure()
    plt.ylim([0, lim_superior])

    x2 = study[0]
    plt.grid(True)
    # plt.figure(figsize=(3,4))

    # "Abrindo" espaço para o xlabel (padding)
    plt.gcf().subplots_adjust(bottom=0.25)

    # Mudando o tamanho do gráfico
    # x = 14
    # y = (x * 9) / 16
    # plt.figure(figsize=(x,y))

    # plt.rcParams['figure.figsize'] = [16,9]
    plt.plot(x2, study[3][:], color='blue', linewidth=2.5)
    plt.errorbar(x2, study[3][:], yerr=err, fmt='o', capsize=2, color='orange', linewidth=1)
    plt.xticks(x2, study[0][:])
    plt.xticks(rotation=90)
    ax = plt.gca()
    # for label in ax.get_xaxis().get_ticklabels()[::2]:
    # label.set_visible(False)
    # plt.xticks(np.arange(0,20, 1))
    plt.title('AUC x Training fold set size - {}'.format(title_graph_rede))
    plt.ylabel('Area under curve (AUC)')
    plt.xlabel('Train test size')
    plt.savefig('AUCxSize_{}_version_{}.png'.format(name_file_rede, version))


# Define o limite superior dos gráficos
def highest_integer(lst, num_epochs):
    c = 1
    for i in range(0, num_epochs, 1):
        if lst[i] > c:
            c = int(round(lst[i]))
    return c

def ultimate_ROC(lauc, auc_all, thres, tpr_all, fpr_all, name_file_rede, title_graph_rede, k_folds, train_size, rede, version):
    print('\n ** Generating ultimate ROC graph for %s...' % rede)
    medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

    mauc = np.percentile(lauc, 50.0)
    mAUCall = np.percentile(auc_all, 50.0)
    print(" ** len tpr_all:", len(tpr_all))
    print(" ** len fpr_all:", len(fpr_all))
    print(" ** lauc:", lauc)
    print(" ** auc_all:", auc_all)

    for num in range(0, int(thres), 1):
        lis = [item for item in tpr_all]
        los = [item for item in fpr_all]

        medians_x.append(np.percentile(los, 50.0))
        medians_y.append(np.percentile(lis, 50.0))
        lowlim.append(np.percentile(lis, 15.87))
        highlim.append(np.percentile(lis, 84.13))

    lowauc = metrics.auc(medians_x, lowlim)
    highauc = metrics.auc(medians_x, highlim)
    print('\n\n\n ** IS THIS CORRECT?')
    print(lowauc, mauc, highauc)
    print(lowauc, mAUCall, highauc)

    # Plotting Final ROC graph
    final_roc_graph(k_folds, train_size, medians_x, medians_y, mauc, lowlim, highlim, name_file_rede, title_graph_rede, version)

    return highauc, lowauc, mauc

def ultimate_fscore(lauc, auc_all, thres, tpr_all, fpr_all, name_file_rede, title_graph_rede, k_folds, train_size, rede, version):
    print('\n ** Generating ultimate ROC graph for %s...' % rede)
    medians_y, medians_x, lowlim, highlim = ([] for i in range(4))

    print(" ** len tpr_all:", len(tpr_all))
    print(" ** len fpr_all:", len(fpr_all))
    print(" ** lauc:", lauc)
    print(" ** auc_all:", auc_all)

    mauc = np.percentile(lauc, 50.0)
    mAUCall = np.percentile(auc_all, 50.0)

    for num in range(0, int(thres), 1):
        los = [item for item in tpr_all]
        lis = [item for item in fpr_all]

        medians_x.append(np.percentile(los, 50.0))
        medians_y.append(np.percentile(lis, 50.0))
        lowlim.append(np.percentile(lis, 15.87))
        highlim.append(np.percentile(lis, 84.13))

    lowauc = metrics.auc(medians_x, lowlim)
    highauc = metrics.auc(medians_x, highlim)
    print('\n\n\n ** IS THIS CORRECT?')
    print(lowauc, mauc, highauc)
    print(lowauc, mAUCall, highauc)

    # Plotting Final ROC graph
    final_fscore_graph(k_folds, train_size, medians_x, medians_y, mauc, lowlim, highlim, name_file_rede, title_graph_rede, version)

    return highauc, lowauc, mauc

def final_fscore_graph(k_folds, train_size, medians_x, medians_y, mauc, lowlim, highlim, name_file_rede, title_graph_rede, version):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.title('Median FScore {} over {} folds with {} training samples'.format(title_graph_rede, k_folds, train_size))
    plt.ylabel('precision', fontsize=14)
    plt.xlabel('recall', fontsize=14)
    plt.plot(medians_x, medians_y, 'b', label='AUC: {}'.format(mauc), linewidth=3)
    plt.fill_between(medians_x, medians_y, lowlim, color='blue', alpha=0.3, interpolate=True)
    plt.fill_between(medians_x, highlim, medians_y, color='blue', alpha=0.3, interpolate=True)
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.savefig('FScoreGraph_{}_Full_{}_version_{}.png'.format(name_file_rede, train_size, version))