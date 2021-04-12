from keras import backend as keras_back
from keras.layers import Input
import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tempfile

import efficientnet.keras as efn
from keras.applications.resnet50 import ResNet50
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
import bisect

from utils import utils, dnn
    
def alternative_model_builder(XTrain, YTrain, architecture, model, bayesian, loss):
    model, model_info = dnn.build_model(XTrain, YTrain,
                                architecture=architecture,
                                name=architecture,
                                bayesian=bayesian,
                                loss=loss,
                                last_activations={'softmax'})
    return model

def model_builder(ix, x_data, weights, avgpool, dropout, preload_weights, loss_regularization):
    keras_back.set_image_data_format('channels_last')
    img_shape = x_data[0,:,:,:].shape
    print("Input Shape Matrix: ", img_shape)
    img_input = Input(shape=img_shape)

    print('\n ** Building network: ', ix)

    model = utils.get_model_roulette(ix, img_shape, img_input, weights)
    #model._name = 'Modified_Model'

    # flat = tf.keras.layers.Flatten()(res_net.output)
    activation = 'relu'
    #def dropout_layer_factory():
    #    return tf.keras.layers.Dropout(rate=0.3, name='dropout')(model.output)
    #model = insert_layer_nonseq(model, '.*activation.*', dropout_layer_factory)

    if avgpool:
        if avgpool and dropout:
            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
            drop_out = tf.keras.layers.Dropout(0.5)(avg_pool)
            dense = tf.keras.layers.Dense(1024, activation='relu')(drop_out)
            drop_out2 = tf.keras.layers.Dropout(0.5)(dense)
            y_hat = tf.keras.layers.Dense(2, activation=activation)(drop_out)
            model = tf.keras.models.Model(model.input, y_hat)
        elif avgpool:
            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
            y_hat = tf.keras.layers.Dense(2, activation=activation)(avg_pool)
            model = tf.keras.models.Model(model.input, y_hat)
    elif dropout:
        drop_out = tf.keras.layers.Dropout(0.7)(model.output)
        y_hat = tf.keras.layers.Dense(2, activation=activation)(drop_out)
        model = tf.keras.models.Model(model.input, y_hat)
    else:
        model = tf.keras.models.Model(model.input)

    if loss_regularization:
        model = add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001))

    #model.save('temp.h5')
    #model = tf.keras.models.load_model('temp.h5')

    if preload_weights:
        model.load_weights(r'./Train_model_weights_%s_50_C1_Backup.h5' % (ix))
    print(" ** Is it true that we're using pretrained weights? ", preload_weights)

    return model

def get_model_effnet(img_shape, img_input, weights, effnet_version):

    if effnet_version == 'B0':
        effnet = efn.EfficientNetB0(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B1':
        effnet = efn.EfficientNetB1(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B2':
        effnet = efn.EfficientNetB2(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B3':
        effnet = efn.EfficientNetB3(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B4':
        effnet = efn.EfficientNetB4(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B5':
        effnet = efn.EfficientNetB5(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    elif effnet_version == 'B6':
        effnet = efn.EfficientNetB6(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)
    else:
        effnet = efn.EfficientNetB7(include_top=False, input_tensor=img_input, weights=weights, pooling=None, input_shape=img_shape)

    return effnet

def compile_model(name_file_rede, model, opt, fold, version, learning_rate, loss, ch1_weights):
    print('\n Compilando rede: ', name_file_rede)
    opt = utils.select_optimizer(opt, learning_rate)

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    print('\n ** Plotting model and callbacks...')
    name_weights = 'Train_model_weights_%s_{epoch:02d}_%s.h5' % (name_file_rede, version)
    if ch1_weights:
        name_weights = 'Train_model_weights_%s_{epoch:02d}_%s_Backup.h5' % (name_file_rede, version)
    csv_name = 'training_{}_fold_{}_ver_{}.csv'.format(name_file_rede, fold, version)
    callbacks = utils.get_callbacks(name_weights=name_weights, patience_lr=10, name_csv=csv_name)

    return callbacks


def fit_model(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks):
    #model.fit_generator(
    history = model.fit(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(x_val_cv, y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks)

    return history

def get_model_resnet(img_shape, img_input, weights, resnet_depth):
    if resnet_depth == 50:
        return ResNet50(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                           pooling=None)
    elif resnet_depth == 101:
        return ResNet101V2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)
    else:
        return ResNet152V2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)

def get_model_inception(img_shape, img_input, weights, version):
    if version == 'V2':
        return InceptionResNetV2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)
    elif version == 'V3':
        return InceptionV3(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)

def get_model_xception(img_shape, img_input, weights):
    return Xception(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape, pooling=None)    

def add_regularization(model, regularizer=tf.keras.regularizers.l2(0.0001)):

    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
      print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
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
    model = tf.keras.models.model_from_json(model_json)
    
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
        model = tf.keras.models.model_from_json(model_json)
    
        # Reload the model weights
        model.load_weights(tmp_weights_path, by_name=True)

    return model
    #return tf.keras.models.Model(inputs=model.inputs, outputs=model_outputs)