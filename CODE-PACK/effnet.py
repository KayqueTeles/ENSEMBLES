from keras import backend as keras_back
from keras.layers import Input
import tensorflow as tf

import efficientnet.keras as efn

from utils.utils import get_callbacks


def get_model_effnet(x_data, weights, effnet_version):
    keras_back.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    print("Input Shape Matrix: ", img_shape)
    # print("X_data Shape: 1- {}, 2- {}, 3- {}".format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    img_input = Input(shape=img_shape)

    print('\n ** Utilizando a Rede EfficientNet ', effnet_version)
    print('\n ** Pesos carregados: ', weights)

    if effnet_version == 'B0':
        effnet = efn.EfficientNetB0(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B1':
        effnet = efn.EfficientNetB1(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B2':
        effnet = efn.EfficientNetB2(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B3':
        effnet = efn.EfficientNetB3(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B4':
        effnet = efn.EfficientNetB4(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B5':
        effnet = efn.EfficientNetB5(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    elif effnet_version == 'B6':
        effnet = efn.EfficientNetB6(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    else:
        effnet = efn.EfficientNetB7(include_top=False, input_tensor=img_input, weights=weights, pooling=None,
                                    input_shape=img_shape)

    # flat = tf.keras.layers.Flatten()(res_net.output)
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(effnet.output)
    # dropout = tf.keras.layers.Dropout(0.5)(avg_pool)
    y_hat = tf.keras.layers.Dense(2, activation="sigmoid")(avg_pool)
    model = tf.keras.models.Model(effnet.input, y_hat)

    return model

def compile_model_effnet(name_file_rede, model, opt, fold, version):
    print('\n Compilando rede: ', name_file_rede)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print('\n ** Plotting model and callbacks...')
    name_weights = 'Train_model_weights_%s_{epoch:02d}_%s.h5' % (name_file_rede, version)
    csv_name = 'training_{}_fold_{}_ver_{}.csv'.format(name_file_rede, fold, version)
    callbacks = get_callbacks(name_weights=name_weights, patience_lr=10, name_csv=csv_name)

    return callbacks


def fit_model_effnet(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks):
    history = model.fit_generator(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(x_val_cv, y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks)

    return history

