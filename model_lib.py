from keras import backend as keras_back
from keras.layers import Input
import tensorflow as tf
import os

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

from utils import utils
    
def model_builder(ix, x_data, weights, avgpool, dropout, preload_weights):
    keras_back.set_image_data_format('channels_last')
    img_shape = x_data[0,:,:,:].shape
    print("Input Shape Matrix: ", img_shape)
    img_input = Input(shape=img_shape)

    print('\n ** Building network: ', ix)
    cwd = os.getcwd()
    print(cwd)
    #if preload_weights:
    #    weights = './Train_model_weights_%s_50_C1_Backup.h5' % (ix)
    #print('\n ** Pesos carregados: ', weights)

    model = utils.get_model_roulette(ix, img_shape, img_input, weights)

    # flat = tf.keras.layers.Flatten()(res_net.output)
    if avgpool:
        if avgpool and dropout:
            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
            drop_out = tf.keras.layers.Dropout(0.5)(avg_pool)
            y_hat = tf.keras.layers.Dense(2, activation="sigmoid")(drop_out)
            model = tf.keras.models.Model(model.input, y_hat)
        elif avgpool:
            avg_pool = tf.keras.layers.GlobalAveragePooling2D()(model.output)
            y_hat = tf.keras.layers.Dense(2, activation="sigmoid")(avg_pool)
            model = tf.keras.models.Model(model.input, y_hat)
    elif dropout:
        drop_out = tf.keras.layers.Dropout(0.5)(model.output)
        y_hat = tf.keras.layers.Dense(2, activation="sigmoid")(drop_out)
        model = tf.keras.models.Model(model.input, y_hat)
    else:
        model = tf.keras.models.Model(model.input)

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