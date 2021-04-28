from keras import backend as keras_back
from keras.layers import Input, Dense, Concatenate, Flatten
import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tempfile
from keras.utils import to_categorical
from keras.layers.merge import concatenate
import keras

import efficientnet_2.keras as efn
from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152
# Todo - Testar a resnet 50 V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from keras.applications import InceptionV3, InceptionResNetV2, Xception
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.optimizers import SGD
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
from IPython.display import Image
from keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import roc_auc_score
from PIL import Image
import matplotlib.pyplot as plt
import bisect
from icecream import ic

from utils import utils, dnn
    
def alternative_model_builder(XTrain, YTrain, architecture, model, bayesian, loss):
    model, model_info = dnn.build_model(XTrain, YTrain,
                                architecture=architecture,
                                name=architecture,
                                bayesian=bayesian,
                                loss=loss,
                                last_activations={'softmax'})
    return model

def gen_flow_for_two_inputs(X1, X2, y, batch_size, gen):
    genX1 = gen.flow(X1,y, batch_size=batch_size,seed=1)
    genX2 = gen.flow(X2, y, batch_size=batch_size,seed=1)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        #ic(X1i, X2i)
        #Assert arrays are equal - this was for peace of mind, but slows down training
        #np.testing.assert_array_equal(X1i[0],X2i[0])
        yield [X1i[0], X2i[0]], X1i[1]

def concatenation(model, model_vis):
    for layer in model.layers:
        model.get_layer(layer.name).name = layer.name + "_y"
    for layer in model_vis.layers:
        model_vis.get_layer(layer.name).name = layer.name + "_vis"
    concat = Concatenate()([model_vis.layers[-2].output, model.layers[-2].output])
    return concat

def model_builder(ix, x_data, x_data_vis, weights, avgpool, dropout, preload_weights, loss_regularization, num_epochs, version):
    keras_back.set_image_data_format('channels_last')
    img_shape = x_data[0,:,:,:].shape
    img_shape_vis = x_data_vis[0,:,:,:].shape
    
    ic(img_shape)
    ic(img_shape_vis)
    img_input = Input(shape=img_shape)
    #img_input_vis = Input(shape=img_shape_vis)
    img_input_vis = Input((200,200,3))

    print('\n ** Building network: ', ix)

    model = utils.get_model_roulette(ix, img_shape, img_input, weights)
    model_vis = utils.get_model_roulette(ix, img_shape_vis, img_input_vis, weights)
    #model._name = 'Modified_Model'

    # flat = keras.layers.Flatten()(res_net.output)
    activation = 'softmax'
    #def dropout_layer_factory():
    #    return keras.layers.Dropout(rate=0.3, name='dropout')(model.output)
    #model = insert_layer_nonseq(model, '.*activation.*', dropout_layer_factory)
    concat = concatenation(model, model_vis)
    y_hat = Dense(2,activation=activation)(concat)
    modelo = keras.models.Model([model_vis.input, model.input], y_hat)

    if loss_regularization:
        model = add_regularization(model, regularizer=keras.regularizers.l2(0.0001))

    f_weig = 0
    ic(preload_weights)
    if preload_weights:
        for k in range(num_epochs):
            if os.path.exists('./Train_model_weights_{}_{}_{}.h5'. format(ix, k, version)):
                model.load_weights(r'./Train_model_weights_%s_%s_%s.h5' % (ix, k, version))
                f_weig += 1
        print(" ** Found weights: ", f_weig)
        utils.delete_weights(ix, version)

    return modelo

def get_model_effnet(img_shape, img_input, weights, effnet_version):

    if effnet_version == 'B0':
        effnet = efn.EfficientNetB0(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B1':
        effnet = efn.EfficientNetB1(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B2':
        #effnet = efn.EfficientNetB2(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
        effnet = efn.EfficientNetB2(input_tensor=img_input, weights=weights)
    elif effnet_version == 'B3':
        effnet = efn.EfficientNetB3(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B4':
        #effnet = efn.EfficientNetB4(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
        effnet = efn.EfficientNetB4(input_tensor=img_input, weights=weights)
    elif effnet_version == 'B5':
        effnet = efn.EfficientNetB5(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B6':
        effnet = efn.EfficientNetB6(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    else:
        effnet = efn.EfficientNetB7(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)

    return effnet

import keras.backend as K
def fbeta(y_true, y_pred):
    TP = (K.sum((y_pred * y_true), axis=-1)) / 64
    FP = (K.sum(((1 - y_pred) * y_true), axis=-1)) / 64
    FN = (K.sum((y_pred * (1 - y_true)), axis=-1)) / 64
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fbeta = (1 + 0.001) * precision * recall / ( 0.001 * precision + recall)
    fbeta = 1 - K.mean(fbeta)
    return fbeta

def compile_model(name_file_rede, model, opt, fold, version, learning_rate, loss, ch1_weights, batch_size):
    print('\n ** Compilando rede: ', name_file_rede)
    opt = utils.select_optimizer(opt, learning_rate)
    if loss == 'fbeta':
        model.compile(loss=fbeta, optimizer=opt, metrics=['accuracy'])
    elif opt == 'RAdam':
        #from opt import RAdam
        from keras_radam import RAdam
        model.compile(loss=loss, optimizer=RAdam(), metrics=['accuracy'])
    else:
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    print('\n ** Plotting model and callbacks...')
    name_weights = 'Train_model_weights_%s_{epoch:02d}_%s.h5' % (name_file_rede, version)
    if ch1_weights:
        name_weights = 'Train_model_weights_%s_{epoch:02d}_%s_Backup.h5' % (name_file_rede, version)
    csv_name = 'training_{}_fold_{}_ver_{}.csv'.format(name_file_rede, fold, version)
    callbacks = utils.get_callbacks(name_weights=name_weights, patience_lr=10, name_csv=csv_name)

    return callbacks


def fit_model(model, generator, x_data_cv, x_data_vis_cv, batch_size, num_epochs, x_val_cv, x_val_vis_cv, y_val_cv, callbacks):
    #model.fit_generator(
    history = model.fit(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=([x_val_vis_cv, x_val_cv], y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks)
    

    return history

def get_model_resnet(img_shape, img_input, weights, resnet_depth):
    if resnet_depth == 50:
        return ResNet50(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                           pooling=None)
    elif resnet_depth == 101:
        return ResNet101(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)
    else:
        return ResNet152(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)

def get_model_inception(img_shape, img_input, weights, version):
    if version == 'V2':
        return InceptionResNetV2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)
    elif version == 'V3':
        return InceptionV3(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)

def get_model_xception(img_shape, img_input, weights):
    return Xception(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)    

def add_regularization(model, regularizer=keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of keras.regularizers.Regularizer")
      return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
              setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # load the model from the config
    model = keras.models.model_from_json(model_json)
    
    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)
    return model

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):
    import re

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = model.to_json()

    # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
        model.save_weights(tmp_weights_path)

        # load the model from the config
        model = keras.models.model_from_json(model_json)
    
        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)

    return model
    #return keras.models.Model(inputs=model.inputs, outputs=model_outputs)

def define_stacked_model(members, opt):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        #print(' starting update loop for network: %s ' % model)
        la = 0
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            la += 1
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + la + '_' + layer.name
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
    model = keras.models.Model(inputs=[model.input for model in members], outputs=output)
    # plot graph of ensemble
    #plot_model(model, show_shapes=True, to_file='model_graph.png')
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

def integrated_ROC(y_test, probs, train_size, model_list, version):
    print(len(probs))
    print('\n ** probs_ensemble: ', probs)

    probsp = probs[:, 1]
    #print('\n ** probsp: ', probsp)
    # print('\n ** probsp.shape: ', probsp.shape)
    y_new = y_test[:, 1]

    thres = 1000

    threshold_v = np.linspace(1, 0, thres)
    tpr, fpr = ([] for i in range(2))

    for tt in range(len(threshold_v)):
        thresh = threshold_v[tt]
        tp_score, fp_score, tn_score, fn_score = (0 for i in range(4))
        for xz in range(len(probsp)):
            if probsp[xz] > thresh:
                if y_new[xz] == 1:
                    tp_score = tp_score + 1
                else:
                    fp_score = fp_score + 1
            else:
                if y_new[xz] == 0:
                    tn_score = tn_score + 1
                else:
                    fn_score = fn_score + 1
        tp_rate = tp_score / (tp_score + fn_score)
        fp_rate = fp_score / (fp_score + tn_score)
        tpr.append(tp_rate)
        fpr.append(fp_rate)

    auc = metrics.auc(fpr, tpr)
    net = 'ensembles'

    print(' ** Plotting %s ROC Graph' % net)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')  # k = color black
    plt.plot(fpr, tpr, label=' AUC: %.3f' % auc, linewidth=3)  # for color 'C'+str(fold), for fold[0 9]
    plt.title('Ensemble ROC over {} models with {} training samples'.format(len(model_list), train_size))
    plt.xlabel('false positive rate', fontsize=14)
    plt.ylabel('true positive rate', fontsize=14)
    plt.legend(loc='lower right', ncol=1, mode='expand')
    plt.savefig('ROCLensDetectNet_{}_Full_{}_version_{}.png'.format('ensembles', train_size, version))

def integrated_stacked_model(models, testX, test_vis, testy, opt, train_size, model_list, version):
    print("\n ** Initiating Stacked Model...")
    members = models
    print(" Models: \n %s" % members)
    stacked_model = define_stacked_model(members, opt)
    print(" Stacked_model: %s" % stacked_model)

    data_x = [testX for _ in range(len(stacked_model.input))]
    print(len(data_x))
    #testy = to_categorical(testy)
    stacked_model.fit(data_x, testy, epochs=50, verbose=0)
    
    yhat = predict_stacked_model(stacked_model, testX)
    #yhat = argmax(yhat, axis=1)
    integrated_ROC(testy, yhat, train_size, model_list, version)
    AUC = roc_auc_score(testy[:, 1], yhat[:, 1])
    print('Integrated Stacked network AUC: %.3f' % AUC)

#class ensemble_metrics():