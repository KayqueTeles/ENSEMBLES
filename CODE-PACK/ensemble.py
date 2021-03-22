from keras import backend as keras_back
from keras.layers import Input
import tensorflow as tf

from effnet import get_model_effnet
from resnet import get_model_resnet
from utils.utils import get_callbacks
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy import dstack
from sklearn.metrics import roc_auc_score
from keras.models import load_model
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.merge import concatenate
from numpy import argmax


def get_model_ensemble(x_data, weights, resnet_depth, effnet_version):
    # TODO Testando comentar esse trecho..
    # keras_back.set_image_data_format('channels_last')
    # img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    # print("Input Shape Matrix: ", img_shape)
    # print("X_data Shape: 1- {}, 2- {}, 3- {}".format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    # img_input = Input(shape=img_shape)

    print('\n ** Utilizando o Ensemble das redes Resnet{}, EfficientNet{}'.format(resnet_depth, effnet_version))
    print('\n ** Pesos carregados: ', weights)

    # TODO TESTE MUDANDO O x_data

    x_data_len = int((len(x_data))/2)

    x_data_resnet = x_data[:x_data_len, :]
    # np.random.shuffle(x_data)
    # x_data_resnet = x_data
    # np.random.shuffle(x_data)
    x_data_effnet = x_data[x_data_len:, :]
    # x_data_effnet = x_data

    print('\n ** Misturando os dados no Ensemble')
    print('\n ** x_data_resnet: ', x_data_resnet.shape)
    print('\n ** x_data_effnet: ', x_data_effnet.shape)

    # *** Resnet = model_1
    # model_resnet = get_model_resnet(x_data, weights, resnet_depth)
    model_resnet = get_model_resnet(x_data_resnet, weights, resnet_depth)
    print('\n ** Modelo Ensemble Resnet Criado!')
    # print('\n', model_resnet.summary(), '\n\n')

    # EfficientNet = model_2
    # model_effnet = get_model_effnet(x_data, weights, effnet_version)
    model_effnet = get_model_effnet(x_data_effnet, weights, effnet_version)
    print('\n ** Modelo Ensemble Efficient Net Criado!')
    # print('\n', model_effnet.summary(), '\n\n')

    # models = [model_resnet, model_effnet]
    # TODO INVERTENDO A ORDEM DE TREINAMENTO DAS REDES
    # models = [model_effnet, model_resnet]

    # return models
    return [model_resnet, model_effnet]


def compile_model_ensemble(name_file_rede, model_resnet, model_effnet, opt, fold, version):
    print('\n Compilando rede: ', name_file_rede)

    print('\n ** Plotting model and callbacks Resnet...')
    model_resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    name_weights_resnet = 'Train_model_weights_%s_resnet_{epoch:02d}_%s.h5' % (name_file_rede, version)
    csv_name_resnet = 'training_{}_resnet_fold_{}.csv'.format(name_file_rede, fold)
    callbacks_resnet = get_callbacks(name_weights=name_weights_resnet, patience_lr=10, name_csv=csv_name_resnet)

    print('\n ** Plotting model and callbacks Efficient Net...')
    model_effnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    name_weights_efn = 'Train_model_weights_%s_effnet_{epoch:02d}_%s.h5' % (name_file_rede, version)
    csv_name_efn = 'training_{}_effnet_fold_{}.csv'.format(name_file_rede, fold)
    callbacks_efn = get_callbacks(name_weights=name_weights_efn, patience_lr=10, name_csv=csv_name_efn)

    # TODO INVERTENDO A ORDEM DE TREINAMENTO DAS REDES

    # print('\n ** Plotting model and callbacks Efficient Net...')
    # model_effnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # name_weights_efn = 'Train_model_weights_%s_effnet_{epoch:02d}.h5' % name_file_rede
    # csv_name_efn = 'training_{}_effnet_fold_{}.csv'.format(name_file_rede, fold)
    # callbacks_efn = get_callbacks(name_weights=name_weights_efn, patience_lr=10, name_csv=csv_name_efn)
    #
    # print('\n ** Plotting model and callbacks Resnet...')
    # model_resnet.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # name_weights_resnet = 'Train_model_weights_%s_resnet_{epoch:02d}.h5' % name_file_rede
    # csv_name_resnet = 'training_{}_resnet_fold_{}.csv'.format(name_file_rede, fold)
    # callbacks_resnet = get_callbacks(name_weights=name_weights_resnet, patience_lr=10,
    #                                  name_csv=csv_name_resnet)

    return [callbacks_resnet, callbacks_efn]

def fit_model_ensemble(model_resnet, model_effnet, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks_resnet, callbacks_efn):

    print('\n ** Fit Rede Resnet')
    history_resnet = model_resnet.fit_generator(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(x_val_cv, y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks_resnet)

    print('\n ** Fit Rede EfficientNet')
    history_efn = model_effnet.fit_generator(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(x_val_cv, y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks_efn)
    # history = [history_resnet, history_efn]

    # TODO INVERTENDO A ORDEM DE TREINAMENTO DAS REDES

    # print('\n ** Fit Rede EfficientNet')
    # history_efn = model_effnet.fit_generator(
    #     generator,
    #     steps_per_epoch=len(x_data_cv) / batch_size,
    #     epochs=num_epochs,
    #     verbose=1,
    #     validation_data=(x_val_cv, y_val_cv),
    #     validation_steps=len(x_val_cv) / batch_size,
    #     callbacks=callbacks_efn)
    #
    # print('\n ** Fit Rede Resnet')
    # history_resnet = model_resnet.fit_generator(
    #     generator,
    #     steps_per_epoch=len(x_data_cv) / batch_size,
    #     epochs=num_epochs,
    #     verbose=1,
    #     validation_data=(x_val_cv, y_val_cv),
    #     validation_steps=len(x_val_cv) / batch_size,
    #     callbacks=callbacks_resnet)

    return [history_resnet, history_efn]

def stacked_logisticregression_ensemble(model_resnet, model_effnet, testX, testy):
    members = [model_resnet, model_effnet]
    en_model = fit_stacked_model(members, testX, testy)
    yhat = stacked_prediction(members, en_model, testX)
    score = roc_auc_score(testy, yhat)
    print(' Stacked models AUC: ', score)
    return score

def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# fit standalone model
	model = LogisticRegression()
	model.fit(stackedX, inputy)
	return model

def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat

def define_stacked_model(members, opt):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        #print(' starting update loop for network: %s ' % model)
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
            #print(" layer: ", layer)
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    print(" ensemble_visible: ", ensemble_visible)
    #concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    print(" ensemble_outputs: ", ensemble_outputs)
    merge = concatenate(ensemble_outputs)
    #print(" model output: ", model.output)
    #print(" model input: ", model.input)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(2, activation='softmax')(hidden)
    model = Model(inputs=[model.input for model in members], outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def fit_stack_model(model, inputX, inputy):
    # prepare input data
    data_x = [inputX for _ in range(len(model.input))]
    print(data_x.shape)
    # encode  output data
    inputy_enc = to_categorical(inputy)
    #print(inputX.shape)
    #print(inputy.shape)
    #print(inputy_enc.shape)
    #print(X.shape)
    #print(X)
    #inputy_enc = inputy
    #history = model.fit_generator(generator, steps_per_epoch=len(x_data_cv) / batch_size, epochs=num_epochs,
        #verbose=1, validation_data=(x_val_cv, y_val_cv), validation_steps=len(x_val_cv) / batch_size)
    # fit model
    model.fit(data_x, inputy_enc, epochs=300)#, verbose=0)
    
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)

def integrated_stacked_model(model_resnet, model_effnet, testX, testy, opt):
    print("\n ** Initiating Stacked Model...")
    members = [model_resnet, model_effnet]
    print(" Models: \n %s" % members)
    stacked_model = define_stacked_model(members, opt)
    print(" Stacked_model: %s" % stacked_model)

    data_x = [testX for _ in range(len(stacked_model.input))]
    print(len(data_x))
    #testy = to_categorical(testy)
    stacked_model.fit(data_x, testy, epochs=50, verbose=0)
    
    yhat = predict_stacked_model(stacked_model, testX)
    #yhat = argmax(yhat, axis=1)
    AUC = roc_auc_score(testy[:, 1], yhat[:, 1])
    print('Integrated Stacked network AUC: %.3f' % AUC)
    return AUC

#class ensemble_metrics():
