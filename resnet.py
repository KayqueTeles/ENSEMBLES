from keras import backend as keras_back
from keras.layers import Input
import tensorflow as tf

from keras.applications.resnet50 import ResNet50
# Todo - Testar a resnet 50 V2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2

from utils.utils import get_callbacks


def get_model_resnet(x_data, weights, resnet_depth):
    keras_back.set_image_data_format('channels_last')
    img_shape = (x_data.shape[1], x_data.shape[2], x_data.shape[3])
    print('Input Shape Matrix: ', img_shape)
    # print('X_data Shape: 1- {}, 2- {}, 3- {}'.format(x_data.shape[1], x_data.shape[2], x_data.shape[3]))
    img_input = Input(shape=img_shape)

    print('\n ** Utilizando a Rede Resnet', resnet_depth)
    print('\n ** Pesos carregados: ', weights)

    if resnet_depth == 50:

        res_net = ResNet50(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                           pooling=None)
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(res_net.output)
        # dropout = tf.keras.layers.Dropout(0.5)(flat)
        # y_hat = tf.keras.layers.Dense(2, activation='sigmoid')(avg_pool)
        y_hat = tf.keras.layers.Dense(2, activation='softmax')(avg_pool)
        model = tf.keras.models.Model(res_net.input, y_hat)

    elif resnet_depth == 101:
        res_net = ResNet101V2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(res_net.output)
        # dropout = tf.keras.layers.Dropout(0.5)(flat)
        # y_hat = tf.keras.layers.Dense(2, activation='sigmoid')(avg_pool)
        y_hat = tf.keras.layers.Dense(2, activation='softmax')(avg_pool)
        model = tf.keras.models.Model(res_net.input, y_hat)

    else:
        res_net = ResNet152V2(include_top=False, weights=weights, input_tensor=img_input, input_shape=img_shape,
                              pooling=None)
        avg_pool = tf.keras.layers.GlobalAveragePooling2D()(res_net.output)
        # dropout = tf.keras.layers.Dropout(0.5)(flat)
        # y_hat = tf.keras.layers.Dense(2, activation='sigmoid')(avg_pool)
        y_hat = tf.keras.layers.Dense(2, activation='softmax')(avg_pool)
        model = tf.keras.models.Model(res_net.input, y_hat)

    return model


def compile_model_resnet(name_file_rede, model, opt, fold, version):
    print('\n Compilando rede: ', name_file_rede)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    print('\n ** Plotting model and callbacks...')
    name_weights = 'Train_model_weights_%s_{epoch:02d}_%s.h5' % (name_file_rede, version)
    csv_name = 'training_{}_fold_{}_ver_{}.csv'.format(name_file_rede, fold, version)
    callbacks = get_callbacks(name_weights=name_weights, patience_lr=10, name_csv=csv_name)

    return callbacks


def fit_model_resnet(model, generator, x_data_cv, batch_size, num_epochs, x_val_cv, y_val_cv, callbacks):
    history = model.fit_generator(
        generator,
        steps_per_epoch=len(x_data_cv) / batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(x_val_cv, y_val_cv),
        validation_steps=len(x_val_cv) / batch_size,
        callbacks=callbacks)

    return history
