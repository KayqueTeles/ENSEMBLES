#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:21:27 2019

@author: manu
"""

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.regularizers import l2
from keras.layers import (Layer, 
                          Input, Dense, Dropout, Lambda, BatchNormalization, 
                          Flatten, Add, Concatenate, 
                          Activation, LeakyReLU,
                          Conv1D, Conv2D, Conv3D,
                          Conv2DTranspose, Conv3DTranspose,
                          SeparableConv1D, SeparableConv2D, 
                          MaxPooling1D, MaxPooling2D, MaxPooling3D, 
                          GlobalAveragePooling1D, GlobalAveragePooling2D, 
                          GlobalAveragePooling3D)

""" Heteroscedastic loss """
def heteroscedastic_loss(true, pred):
    mean = pred[:, :1]
    log_var = pred[:, 1:]
    precision = K.exp(-log_var)
    return K.sum(precision * (true-mean)**2. + log_var, -1)

def challenge_loss(true, pred):

    P,_ = tf.metrics.precision(true, pred)
    R,_ = tf.metrics.recall(true, pred)

    beta2 = tf.constant(.001, dtype = tf.float32)

    FB = (1+beta2)*(P*R)/(beta2*P + R)

    return -FB


""" Conv1D transpose """
def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

""" We'll use this snippet to calculate the maximum number of convolution 
        blocks (inception, resnet, etc) that we can apply to our data """
def pool_outshape(sz,k,s,times=1):
    def _get_shape(sz,k,s):
        return (sz-k)//s + 1
    for _ in range(times):
        sz = _get_shape(sz,k,s)
    return sz

l2_reg = 0.0001

""" Output dense tree (common to a lot of networks)"""
def output_dense_tree(mid_stream, output_shape, last_activations, 
                      heteroscedastic = False, units = [512, 256, 128, 64]):
    outputs = []
    for oname in output_shape:
				
        branchtmp = mid_stream
        for u in units:
            branchtmp = Dense(u, kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(branchtmp)
            branchtmp = BatchNormalization()(branchtmp)
            branchtmp = LeakyReLU()(branchtmp)
            #branchtmp = Activation('relu')(branchtmp)
        
        if not heteroscedastic:
            out = Dense(output_shape[oname][1],
                        kernel_initializer='normal',
                        activation = last_activations[oname],
                        name = oname)(branchtmp)
        else:
            out_mean = Dense(output_shape[oname][1],
                             kernel_initializer = 'normal',
                             activation = last_activations[oname],
                             name = oname+'_mean')
                             #activation = last_activations[oname],
            
            out_log_var = Dense(output_shape[oname][1],
                             kernel_initializer = 'normal',
                             activation = last_activations[oname],
                             name = oname+'_logvar')(branchtmp)
            
            out = Concatenate(axis=1,name = oname)([out_mean, out_log_var])
            
        
        outputs.append(out)
    
    return outputs


""" Residual blocks """
# ResNetXt as seen in here: https://blog.waya.ai/deep-residual-learning-9610bb62c355
def grouped_convolution(layer, nb_channels, _strides, cardinality = 1):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        if len(layer.get_shape().as_list()[1:]) == 4:
            return Conv3D(nb_channels, kernel_size = (3, 3, 3), 
                          strides = _strides, padding = 'same')(layer)
        elif len(layer.get_shape().as_list()[1:]) == 3:
            return Conv2D(nb_channels, kernel_size = (3, 3), 
                          strides = _strides, padding = 'same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(layer)
        else:
            return Conv1D(nb_channels, kernel_size = 3, 
                          strides = _strides, padding = 'same')(layer)
    
    assert not nb_channels % cardinality
    _d = nb_channels // cardinality

    # in a grouped convolution layer, input and output channels are divided into 
    # `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        if len(layer.get_shape().as_list()[1:]) == 4:
            group = Lambda(lambda z: z[:, :, :, :, j * _d:j * _d + _d])(layer)
            groups.append(Conv2D(_d, kernel_size=(3, 3, 3), strides = _strides, 
                                 padding = 'same')(group))
        elif len(layer.get_shape().as_list()[1:]) == 3:
            group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(layer)
            groups.append(Conv2D(_d, kernel_size=(3, 3), strides = _strides, 
                                 padding = 'same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(group))
        else:
            group = Lambda(lambda z: z[:, :, j * _d:j * _d + _d])(layer)
            groups.append(Conv1D(_d, kernel_size = 3, strides = _strides, 
                                 padding = 'same')(group))
        
    # the grouped convolutional layer concatenates them as the outputs of the layer
    layer = Concatenate()(groups)

    return layer    
    
def residual_block(y, nb_channels_in, nb_channels_out, 
                   cardinality = 1, _strides = (1, 1, 1), 
                   _project_shortcut = False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y

    """ Image processing residual block """
    if len(y.get_shape().as_list()[1:]) == 4:
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Conv3D(nb_channels_in, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                   padding='same')(y)
        y = LeakyReLU()(y)
        y = BatchNormalization()(y)
        
        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides = _strides,
                                cardinality = cardinality)
        y = LeakyReLU()(y)
        y = BatchNormalization()(y)
        
        y = Conv3D(nb_channels_out, kernel_size=(1, 1, 1), strides=(1, 1, 1), 
                   padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = BatchNormalization()(y)
    
        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv3D(nb_channels_out, kernel_size = (1, 1, 1), 
                              strides=_strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
            
    if len(y.get_shape().as_list()[1:]) == 3:
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Dropout(.3)(y)
        y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), 
                   padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(y)
        y = LeakyReLU()(y)
        y = BatchNormalization()(y)
        
    
        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides = _strides[:2], 
                                cardinality = cardinality)
        y = LeakyReLU()(y)
        y = BatchNormalization()(y)
        
    
        y = Conv2D(nb_channels_out, kernel_size = (1, 1), strides = (1, 1), 
                   padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = BatchNormalization()(y)
    
        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            _strides = (1, 1) # add Luciana
            shortcut = Conv2D(nb_channels_out, kernel_size = (1, 1),
                              strides = _strides, padding = 'same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(shortcut)
            shortcut = BatchNormalization()(shortcut)
    
    else:
        # we modify the residual building block as a bottleneck design to make the network more economical
        y = Conv1D(nb_channels_in, kernel_size = 1, strides = 1, 
                   padding = 'same')(y)
        y = LeakyReLU()(y)
        y = BatchNormalization()(y)
        
        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides = _strides[0],
                                cardinality = cardinality)
        y = LeakyReLU()(y)
        y = BatchNormalization()(y)
        
        y = Conv1D(nb_channels_out, kernel_size = 1, strides = 1, 
                   padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = BatchNormalization()(y)
    
        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = Conv1D(nb_channels_out, kernel_size = 1, 
                              strides = _strides[0], padding = 'same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
            
    y = Add()([shortcut, y])

    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = LeakyReLU()(y)

    return y


"""
Custom Keras Layers
"""
class Tile(Layer):
    def __init__(self, reps, **kwargs):
        self.reps = reps
        super(Tile, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Tile, self).build(input_shape)
    
    def call(self, x):
        y = K.tile(x, (1,) + self.reps)
        return y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + tuple(np.array(self.reps)*np.array(input_shape[1:]))

"""
Architectures
"""


""" Patrick Net """
class PatrickNet(object):
    def __init__(self, input_shape, output_shape, last_activations = None, heteroscedastic = False, 
                name = 'patricknet', activation = 'elu', **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        self.input_type = x_type
        self.activation = activaton
        
    def Residual(self, filters, out, activation, skip, **kwargs):
        conv1 = Conv2D(filters, (3,3), padding="same")(skip)
        bn1_1 = BatchNormalization()(conv1)
        act1 = Activation(activation)(bn1_1)
        bn1_2 = BatchNormalization()(act1)
        conv2 = Conv2D(filters, (3,3), padding="same")(bn1_2)
        bn2_1 = BatchNormalization()(conv2)
        act2 = Activation(activation)(bn2_1)
        bn2_2 = BatchNormalization()(act2)
        conv3 = Conv2D(filters, (3,3), padding="same")(bn2_2)
        bn3_1 = BatchNormalization()(conv3)
        act3 = Activation(activation)(bn3_1)
        bn3_2 = BatchNormalization()(act3)
        add = Add()([skip, bn3_2])
        bn_add = BatchNormalization()(add)
        mp = MaxPooling2D((2,2))(bn_add)
        bn_mp = BatchNormalization()(mp)
        out_conv = Conv2D(out, (1,1), padding="same")(bn_mp)
        out_bn = BatchNormalization()(out_conv)
        return out_bn
        
    def build(self, **kwargs):
        
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'image':
                res1 = self.Residual(64, 128, self.activation,inputLayer)
                res2 = self.Residual(128, 256, self.activation,res1)
                res3 = self.Residual(256, 512,self.activation,res2)
                res4 = self.Residual(512, 1,self.activation,res3)
                layer = Flatten()(res4)
            
            inputStreams.append(layer)
        
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
        
        """ Output stream """
        outputs = []
        for oname in self.output_shape:
            
            
            dense1 = Dense(144, activation=self.activation)(mid_stream)
            bn_dense1 = BatchNormalization()(dense1)
            dense2 = Dense(144, activation=self.activation)(bn_dense1)
            bn_dense2 = BatchNormalization()(dense2)
            if not self.heteroscedastic:
                out = Dense(self.output_shape[oname][1],
                            activation = self.last_activations[oname],
                            name = oname)(bn_dense2)
            else:
                out_mean = Dense(self.output_shape[oname][1],
                                 activation = self.last_activations[oname],
                                 name = oname)(bn_dense2)
                
                out_log_var = Dense(self.output_shape[oname][1],
                                 activation = self.last_activations[oname],
                                 name = oname)(bn_dense2)
                
                out = Concatenate(axis=1)([out_mean, out_log_var])
            
            outputs.append(out)
        
        return inputs, outputs, mid_stream



""" DUMMYNET """
class DummyNet(object):
    def __init__(self, input_shape, output_shape, 
                 last_activations = None, heteroscedastic = False, 
                 name = 'dummynet', **kwargs):
        
        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        self.input_type = x_type
        
    """ Build dummy network """
    def build(self, **kwargs):
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
                layer = Flatten()(inputLayer)
                
            elif self.input_type[iname] == 'image':
                layer = Flatten()(inputLayer)
            
            inputStreams.append(layer)
        
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
        
        """ 
        Output Stream 
        """
        outputs = []
        for oname in self.output_shape:
            branchtmp = Dense(64)(mid_stream)
            branchtmp = LeakyReLU()(branchtmp)
            #branchtmp = Activation('relu')(branchtmp)
            branchtmp = BatchNormalization()(branchtmp)
            
            if not self.heteroscedastic:
                out = Dense(self.output_shape[oname][1],
                            kernel_initializer='normal',
                            activation = self.last_activations[oname],
                            name = oname)(branchtmp)
            else:
                out_mean = Dense(self.output_shape[oname][1],
                                 kernel_initializer = 'normal',
                                 activation = self.last_activations[oname],
                                 name = oname)(branchtmp)
                
                out_log_var = Dense(self.output_shape[oname][1],
                                 kernel_initializer = 'normal',
                                 activation = self.last_activations[oname],
                                 name = oname)(branchtmp)
                
                out = Concatenate(axis=1)([out_mean, out_log_var])
            
            outputs.append(out)
        
        return inputs, outputs, mid_stream


""" SEQUENTIAL NET """
class SequentialNet(object):
    def __init__(self, input_shape, output_shape, 
                 last_activations = None, heteroscedastic = False, 
                 name = 'sequentialnet', out_units = [512, 256, 128, 64],
                 **kwargs):
        
        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        self.out_units = out_units
        
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        self.input_type = x_type
        
    """ Build sequential network """
    def build(self, **kwargs):
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
                layer = Conv3D(64,kernel_size=(5,5,5))(inputLayer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer)
                #layer = Activation('relu')(layer)
                layer = MaxPooling3D(pool_size=(2,2,2))(layer)
                
                layer = Conv3D(128,kernel_size=(5,5,5))(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling3D(pool_size=(2,2,2))(layer)
                
                layer = Conv3D(256,kernel_size=(3,3,3))(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling3D(pool_size=(2,2,2))(layer)
                
                layer = Conv3D(512,kernel_size=(3,3,3))(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling3D(pool_size=(2,2,2))(layer)
                layer = Flatten()(layer)
                
            elif self.input_type[iname] == 'image':
                layer = Conv2D(64,kernel_size=(5,5))(inputLayer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling2D(pool_size=(2,2))(layer)
                
                layer = Conv2D(128,kernel_size=(5,5))(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling2D(pool_size=(2,2))(layer)
                
                layer = Conv2D(256,kernel_size=(3,3))(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling2D(pool_size=(2,2))(layer)
                
                layer = Conv2D(512,kernel_size=(3,3))(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling2D(pool_size=(2,2))(layer)
                layer = Flatten()(layer)
                
            elif self.input_type[iname] == 'log':
                layer = Dense(64)(inputLayer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
    				
                layer = Dense(128)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = Dense(256)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = Dense(512)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
            inputStreams.append(layer)
            
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
            
        """ 
        Output Stream 
        """
        outputs = output_dense_tree(mid_stream, 
                                    self.output_shape, 
                                    self.last_activations,
                                    heteroscedastic = self.heteroscedastic,
                                    units = self.out_units)
        
        return inputs, outputs, mid_stream
        
    

""" FLAT NET """
class FlatNet(object):
    def __init__(self, input_shape, output_shape, 
                 last_activations = None, heteroscedastic = False,
                 name = 'flatnet', out_units = [512, 256, 128, 64], **kwargs):
        
        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        self.out_units = out_units
        
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        self.input_type = x_type
        
    """ Build flat network """
    def build(self, **kwargs):
        
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        ush = {xn: np.prod(self.input_shape[xn][1:]) for xn in self.input_shape}
        umax = max([ush[xn] for xn in ush])
        
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
                if ush[iname] != umax:
                    inputLayer = Conv3D(int(self.input_shape[iname][-1]*umax/np.prod(self.input_shape[iname][1:-1])),
                                        (1,1,1), padding = 'same')(inputLayer)
                
                layer = Flatten()(inputLayer)
                
            elif self.input_type[iname] == 'image':
                if ush[iname] != umax:
                    inputLayer = Conv2D(int(self.input_shape[iname][-1]*umax/np.prod(self.input_shape[iname][1:-1])),
                                        (1,1), padding = 'same')(inputLayer)
                
                layer = Flatten()(inputLayer)
                
            elif self.input_type[iname] == 'log':
                layer = inputLayer
                print(self.input_shape)
                if ush[iname] != umax:
                    layer = Conv1D(int(self.input_shape[iname][-1]*umax/np.prod(self.input_shape[iname][1:-1])),\
                                        (1), padding = 'same')(layer)
                
                if len(self.input_shape[iname][1:]) >= 2:
                    layer = Flatten()(layer)
                
            inputStreams.append(layer)
        
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
            
        """ 
        Output Stream 
        """
        outputs = output_dense_tree(mid_stream, 
                                    self.output_shape, 
                                    self.last_activations,
                                    heteroscedastic = self.heteroscedastic,
                                    units = self.out_units)
        
        return inputs, outputs, mid_stream
        

# """ INCEPTION NET """
# class InceptionNet(object):
#     def __init__(self, input_shape, output_shape,
#                  last_activations = None, heteroscedastic = False,
#                  name = 'inceptionnet', out_units = [512, 256, 128, 64], **kwargs):
#
#         """ Set Parameters """
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.name = name
#         if last_activations is None:
#             last_activations = {xn: None for xn in self.input_shape}
#         self.last_activations = last_activations
#         self.heteroscedastic = heteroscedastic
#
#         x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
#                   for xn in input_shape}
#         self.input_type = x_type
#         self.out_units = out_units
#
#     """ Build inception network """
#     def build(self, **kwargs):
#
#         """
#         Input Stream
#         """
#         inputs = []
#         inputStreams = []
#         for iname in self.input_shape:
#             inputLayer = Input(shape = self.input_shape[iname][1:],
#                                name = iname)
#             # add input layer to inputs
#             inputs.append(inputLayer)
#
#             if self.input_type[iname] == 'volume':
#
#                 layer = Conv3D(64,kernel_size=(5,5,5))(inputLayer)
#                 layer = Activation('relu')(layer)
#                 layer = BatchNormalization()(layer)
#                 layer = MaxPooling3D(pool_size=(2,2,2))(layer)
#
#                 """
#                 Core Stream
#                 """
#                 # Calculate maximum number of inception blocks allowed by data size
#                 (H0,W0,D0,_) = layer.get_shape().as_list()[1:]
#
#                 ninceptions_max = np.minimum(int(np.ceil(np.log2(H0))),
#                                              int(np.ceil(np.log2(W0))),
#                                              int(np.ceil(np.log2(D0))))
#
#                 H = np.array([pool_outshape(H0,2,2,times=t) \
#                               for t in np.arange(1,ninceptions_max)])
#                 W = np.array([pool_outshape(W0,2,2,times=t) \
#                               for t in np.arange(1,ninceptions_max)])
#                 D = np.array([pool_outshape(D0,2,2,times=t) \
#                               for t in np.arange(1,ninceptions_max)])
#                 #print(ninceptions_max,H0,W0,H,W)
#                 ninceptions_max = np.maximum(0,
#                                              np.minimum(np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else -1,
#                                                         np.max(np.where(W > 1)[0]) if np.where(W>1)[0] != [] else -1,
#                                                         np.max(np.where(D > 1)[0]) if np.where(D>1)[0] != [] else -1) + 1
#                                              )
#
#                 filters = 32
#                 for icp in range(ninceptions_max):
#                     tower_0 = Conv3D(filters, (1,1,1), padding='same')(layer)
#                     tower_0 = Activation('relu')(tower_0)
#                     tower_0 = BatchNormalization()(tower_0)
#
#                     tower_1 = Conv3D(filters, (1,1,1), padding='same',
#                                      activation='relu')(layer)
#                     tower_1 = Conv3D(filters, (3,3,3), padding='same')(tower_1)
#                     tower_1 = Activation('relu')(tower_1)
#                     tower_1 = BatchNormalization()(tower_1)
#
#                     tower_2 = Conv3D(filters, (1,1,1), padding='same',
#                                      activation='relu')(layer)
#                     tower_2 = Conv3D(filters, (3,3,3), padding='same',
#                                      activation='relu')(tower_2)
#                     tower_2 = Conv3D(filters, (3,3,3), padding='same')(tower_2)
#                     tower_2 = Activation('relu')(tower_2)
#                     tower_2 = BatchNormalization()(tower_2)
#
#                     tower_3 = MaxPooling3D((3,3,3), strides=(1,1,1),
#                                            padding='same')(layer)
#                     tower_3 = Conv3D(filters, (1,1,1), padding='same')(tower_3)
#                     tower_3 = Activation('relu')(tower_3)
#                     tower_3 = BatchNormalization()(tower_3)
#
#                     layer = Concatenate(axis = 4)([tower_0, tower_1, tower_2, tower_3])
#
#                     layer = MaxPooling3D((2,2,2))(layer)
#
#                     #layer = BatchNormalization()(layer)
#                     filters = filters*2
#
#             elif self.input_type[iname] == 'image':
#
#                 layer = Conv2D(64,kernel_size=(5,5))(inputLayer)
#                 layer = Activation('relu')(layer)
#                 layer = BatchNormalization()(layer)
#                 layer = MaxPooling2D(pool_size=(2,2))(layer)
#
#                 """
#                 Core Stream
#                 """
#                 # Calculate maximum number of inception blocks allowed by data size
#                 (H0,W0,_) = layer.get_shape().as_list()[1:]
#
#                 ninceptions_max = np.minimum(int(np.ceil(np.log2(H0))),
#                                              int(np.ceil(np.log2(W0))))
#                 H = np.array([pool_outshape(H0,2,2,times=t) \
#                               for t in np.arange(1,ninceptions_max)])
#                 W = np.array([pool_outshape(W0,2,2,times=t) \
#                               for t in np.arange(1,ninceptions_max)])
#                 #print(ninceptions_max,H0,W0,H,W)
#                 ninceptions_max = np.maximum(0,
#                                              np.minimum(np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else -1,
#                                                         np.max(np.where(W > 1)[0]) if np.where(W>1)[0] != [] else -1) + 1
#                                              )
#
#                 filters = 32
#                 for icp in range(ninceptions_max):
#                     tower_0 = Conv2D(filters, (1,1), padding='same')(layer)
#                     tower_0 = Activation('relu')(tower_0)
#                     tower_0 = BatchNormalization()(tower_0)
#
#                     tower_1 = Conv2D(filters, (1,1), padding='same', activation='relu')(layer)
#                     tower_1 = Conv2D(filters, (3,3), padding='same')(tower_1)
#                     tower_1 = Activation('relu')(tower_1)
#                     tower_1 = BatchNormalization()(tower_1)
#
#                     tower_2 = Conv2D(filters, (1,1), padding='same', activation='relu')(layer)
#                     tower_2 = Conv2D(filters, (3,3), padding='same', activation='relu')(tower_2)
#                     tower_2 = Conv2D(filters, (3,3), padding='same')(tower_2)
#                     tower_2 = Activation('relu')(tower_2)
#                     tower_2 = BatchNormalization()(tower_2)
#
#                     tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer)
#                     tower_3 = Conv2D(filters, (1,1), padding='same')(tower_3)
#                     tower_3 = Activation('relu')(tower_3)
#                     tower_3 = BatchNormalization()(tower_3)
#
#                     layer = Concatenate(axis = 3)([tower_0, tower_1, tower_2, tower_3])
#
#                     layer = MaxPooling2D((2,2))(layer)
#
#                     #layer = BatchNormalization()(layer)
#                     filters = filters*2
#
#             elif self.input_type[iname] == 'log':
#
#                 # Calculate maximum number of inception blocks allowed by data size
#                 (H0,W0) = inputLayer.get_shape().as_list()[1:]
#
#                 ninceptions_max = int(np.ceil(np.log2(H0)))
#
#                 H = np.array([pool_outshape(H0,2,2,times=t) \
#                               for t in np.arange(1,ninceptions_max)])
#
#                 #print(ninceptions_max,H0,W0,H,W)
#                 ninceptions_max = np.maximum(0,
#                                              np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else -1
#                                              )
#
#                 filters = 32
#                 for _ in range(ninceptions_max):
#                     tower_0 = Conv1D(filters, (1), padding='same')(inputLayer)
#                     tower_0 = Activation('relu')(tower_0)
#                     tower_0 = BatchNormalization()(tower_0)
#
#                     tower_1 = Conv1D(filters, (1), padding='same',
#                                      activation='relu')(inputLayer)
#                     tower_1 = Conv1D(filters, (3), padding='same')(tower_1)
#                     tower_1 = Activation('relu')(tower_1)
#                     tower_1 = BatchNormalization()(tower_1)
#
#                     tower_2 = Conv1D(filters, (1), padding='same',
#                                      activation='relu')(inputLayer)
#                     tower_2 = Conv1D(filters, (3), padding='same',
#                                      activation='relu')(tower_2)
#                     tower_2 = Conv1D(filters, (3), padding='same')(tower_2)
#                     tower_2 = Activation('relu')(tower_2)
#                     tower_2 = BatchNormalization()(tower_2)
#
#                     tower_3 = MaxPooling1D(3, strides=1, padding='same')(inputLayer)
#                     tower_3 = Conv1D(filters, (1), padding='same')(tower_3)
#                     tower_3 = Activation('relu')(tower_3)
#                     tower_3 = BatchNormalization()(tower_3)
#
#                     layer = Concatenate(axis = 2)([tower_0, tower_1, tower_2, tower_3])
#
#                     layer = MaxPooling1D(2)(layer)
#
#                     #layer = BatchNormalization()(layer)
#                     filters = filters*2
#
#             # Flatten activations
#             layer = Flatten()(layer)
#
#             # Add stream to input streams
#             inputStreams.append(layer)
#
#         """ Mid stream """
#         mid_stream = layer
#         if len(self.input_shape) > 1:
#             # add concatenate layer
#             mid_stream = Concatenate(axis=1)(inputStreams)
#
#         """
#         Output Stream
#         """
#         outputs = output_dense_tree(mid_stream,
#                                     self.output_shape,
#                                     self.last_activations,
#                                     heteroscedastic = self.heteroscedastic,
#                                     units = self.out_units)
#
#         return inputs, outputs, mid_stream


""" INCEPTION NET """


class InceptionNet(object):
    def __init__(self, input_shape, output_shape,
                 last_activations=None, heteroscedastic=False,
                 name='inceptionnet', out_units=[512, 256, 128, 64], ModeChange=None, **kwargs):

        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic

        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log' \
                  for xn in input_shape}
        self.input_type = x_type
        self.out_units = out_units

        self.ModeChange = ModeChange

    """ Build inception network """

    def build(self, **kwargs):

        """
        Input Stream
        """

        if self.ModeChange is False:
            inputs = []
            inputStreams = []
            for iname in self.input_shape:
                inputLayer = Input(shape=self.input_shape[iname][1:],
                                   name=iname)
                # add input layer to inputs
                inputs.append(inputLayer)

                if self.input_type[iname] == 'volume':

                    layer = Conv3D(64, kernel_size=(5, 5, 5))(inputLayer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

                    """ 
                    Core Stream 
                    """
                    # Calculate maximum number of inception blocks allowed by data size
                    (H0, W0, D0, _) = layer.get_shape().as_list()[1:]

                    ninceptions_max = np.minimum(int(np.ceil(np.log2(H0))),
                                                 int(np.ceil(np.log2(W0))),
                                                 int(np.ceil(np.log2(D0))))

                    H = np.array([pool_outshape(H0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    W = np.array([pool_outshape(W0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    D = np.array([pool_outshape(D0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    # print(ninceptions_max,H0,W0,H,W)
                    ninceptions_max = np.maximum(0,
                                                 np.minimum(np.max(np.where(H > 1)[0]) if np.where(H > 1)[0] != [] else -1,
                                                            np.max(np.where(W > 1)[0]) if np.where(W > 1)[0] != [] else -1,
                                                            np.max(np.where(D > 1)[0]) if np.where(D > 1)[
                                                                                              0] != [] else -1) + 1
                                                 )

                    filters = 32
                    for icp in range(ninceptions_max):
                        tower_0 = Conv3D(filters, (1, 1, 1), padding='same')(layer)
                        tower_0 = LeakyReLU()(tower_0) #tower_0 = Activation('relu')(tower_0)
                        tower_0 = BatchNormalization()(tower_0)

                        tower_1 = Conv3D(filters, (1, 1, 1), padding='same',
                                         activation='relu')(layer)
                        tower_1 = Conv3D(filters, (3, 3, 3), padding='same')(tower_1)
                        tower_1 = LeakyReLU()(tower_1) #tower_1 = Activation('relu')(tower_1)
                        tower_1 = BatchNormalization()(tower_1)

                        tower_2 = Conv3D(filters, (1, 1, 1), padding='same')(layer)
                        tower_2 = Conv3D(filters, (3, 3, 3), padding='same')(tower_2)
                        tower_2 = Conv3D(filters, (3, 3, 3), padding='same')(tower_2)
                        tower_2 = LeakyReLU()(tower_2) #tower_2 = Activation('relu')(tower_2)
                        tower_2 = BatchNormalization()(tower_2)

                        tower_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1),
                                               padding='same')(layer)
                        tower_3 = Conv3D(filters, (1, 1, 1), padding='same')(tower_3)
                        tower_3 = LeakyReLU()(tower_3) #tower_3 = Activation('relu')(tower_3)
                        tower_3 = BatchNormalization()(tower_3)

                        layer = Concatenate(axis=4)([tower_0, tower_1, tower_2, tower_3])

                        layer = MaxPooling3D((2, 2, 2))(layer)

                        # layer = BatchNormalization()(layer)
                        filters = filters * 2

                elif self.input_type[iname] == 'image':

                    layer = Conv2D(64, kernel_size=(5, 5))(inputLayer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    layer = MaxPooling2D(pool_size=(2, 2))(layer)

                    """ 
                    Core Stream 
                    """
                    # Calculate maximum number of inception blocks allowed by data size
                    (H0, W0, _) = layer.get_shape().as_list()[1:]

                    ninceptions_max = np.minimum(int(np.ceil(np.log2(H0))),
                                                 int(np.ceil(np.log2(W0))))
                    H = np.array([pool_outshape(H0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    W = np.array([pool_outshape(W0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    # print(ninceptions_max,H0,W0,H,W)
                    ninceptions_max = np.maximum(0,
                                                 np.minimum(np.max(np.where(H > 1)[0]) if np.where(H > 1)[0] != [] else -1,
                                                            np.max(np.where(W > 1)[0]) if np.where(W > 1)[
                                                                                              0] != [] else -1) + 1
                                                 )

                    filters = 32
                    for icp in range(ninceptions_max):
                        tower_0 = Conv2D(filters, (1, 1), padding='same')(layer)
                        tower_0 = LeakyReLU()(tower_0) #tower_0 = Activation('relu')(tower_0)
                        tower_0 = BatchNormalization()(tower_0)

                        tower_1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(layer)
                        tower_1 = Conv2D(filters, (3, 3), padding='same')(tower_1)
                        tower_1 = LeakyReLU()(tower_1) #tower_1 = Activation('relu')(tower_1)
                        tower_1 = BatchNormalization()(tower_1)

                        tower_2 = Conv2D(filters, (1, 1), padding='same', activation='relu')(layer)
                        tower_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(tower_2)
                        tower_2 = Conv2D(filters, (3, 3), padding='same')(tower_2)
                        tower_2 = LeakyReLU()(tower_2) #tower_2 = Activation('relu')(tower_2)
                        tower_2 = BatchNormalization()(tower_2)

                        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer)
                        tower_3 = Conv2D(filters, (1, 1), padding='same')(tower_3)
                        tower_3 = LeakyReLU()(tower_3) #tower_3 = Activation('relu')(tower_3)
                        tower_3 = BatchNormalization()(tower_3)

                        layer = Concatenate(axis=3)([tower_0, tower_1, tower_2, tower_3])

                        layer = MaxPooling2D((2, 2))(layer)

                        # layer = BatchNormalization()(layer)
                        filters = filters * 2

                elif self.input_type[iname] == 'log':

                    # Calculate maximum number of inception blocks allowed by data size
                    (H0, W0) = inputLayer.get_shape().as_list()[1:]

                    ninceptions_max = int(np.ceil(np.log2(H0)))

                    H = np.array([pool_outshape(H0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])

                    # print(ninceptions_max,H0,W0,H,W)
                    ninceptions_max = np.maximum(0,
                                                 np.max(np.where(H > 1)[0]) if np.where(H > 1)[0] != [] else -1
                                                 )

                    filters = 32
                    for _ in range(ninceptions_max):
                        tower_0 = Conv1D(filters, (1), padding='same')(inputLayer)
                        tower_0 = LeakyReLU()(tower_0) #tower_0 = Activation('relu')(tower_0)
                        tower_0 = BatchNormalization()(tower_0)

                        tower_1 = Conv1D(filters, (1), padding='same')(inputLayer)
                        tower_1 = Conv1D(filters, (3), padding='same')(tower_1)
                        tower_1 = LeakyReLU()(tower_1) #tower_1 = Activation('relu')(tower_1)
                        tower_1 = BatchNormalization()(tower_1)

                        tower_2 = Conv1D(filters, (1), padding='same')(inputLayer)
                        tower_2 = Conv1D(filters, (3), padding='same')(tower_2)
                        tower_2 = Conv1D(filters, (3), padding='same')(tower_2)
                        tower_2 = LeakyReLU()(tower_2) #tower_2 = Activation('relu')(tower_2)
                        tower_2 = BatchNormalization()(tower_2)

                        tower_3 = MaxPooling1D(3, strides=1, padding='same')(inputLayer)
                        tower_3 = Conv1D(filters, (1), padding='same')(tower_3)
                        tower_3 = LeakyReLU()(tower_3) #tower_3 = Activation('relu')(tower_3)
                        tower_3 = BatchNormalization()(tower_3)

                        layer = Concatenate(axis=2)([tower_0, tower_1, tower_2, tower_3])

                        layer = MaxPooling1D(2)(layer)

                        # layer = BatchNormalization()(layer)
                        filters = filters * 2

                # Flatten activations
                layer = Flatten()(layer)

                # Add stream to input streams
                inputStreams.append(layer)

            """ Mid stream """
            mid_stream = layer
            if len(self.input_shape) > 1:
                # add concatenate layer
                mid_stream = Concatenate(axis=1)(inputStreams)

            """ 
            Output Stream 
            """
            outputs = output_dense_tree(mid_stream,
                                        self.output_shape,
                                        self.last_activations,
                                        heteroscedastic=self.heteroscedastic,
                                        units=self.out_units)
        else:
            inputs = []
            inputStreams = []
            for iname in self.input_shape:
                inputLayer = Input(shape=self.input_shape[iname][1:],
                                   name=iname)
                # add input layer to inputs
                inputs.append(inputLayer)

                if self.input_type[iname] == 'volume':

                    layer = Conv3D(64, kernel_size=(5, 5, 5))(inputLayer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    layer = MaxPooling3D(pool_size=(2, 2, 2))(layer)

                    """ 
                    Core Stream 
                    """
                    # Calculate maximum number of inception blocks allowed by data size
                    (H0, W0, D0, _) = layer.get_shape().as_list()[1:]

                    ninceptions_max = np.minimum(int(np.ceil(np.log2(H0))),
                                                 int(np.ceil(np.log2(W0))),
                                                 int(np.ceil(np.log2(D0))))

                    H = np.array([pool_outshape(H0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    W = np.array([pool_outshape(W0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    D = np.array([pool_outshape(D0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    # print(ninceptions_max,H0,W0,H,W)
                    ninceptions_max = np.maximum(0,
                                                 np.minimum(np.max(np.where(H > 1)[0]) if np.where(H > 1)[0] != [] else -1,
                                                            np.max(np.where(W > 1)[0]) if np.where(W > 1)[0] != [] else -1,
                                                            np.max(np.where(D > 1)[0]) if np.where(D > 1)[
                                                                                              0] != [] else -1) + 1
                                                 )

                    filters = 32
                    for icp in range(ninceptions_max):
                        tower_0 = Conv3D(filters, (1, 1, 1), padding='same')(layer)
                        tower_0 = LeakyReLU()(tower_0) #tower_0 = Activation('relu')(tower_0)
                        tower_0 = BatchNormalization()(tower_0)

                        tower_1 = Conv3D(filters, (1, 1, 1), padding='same')(layer)
                        tower_1 = Conv3D(filters, (3, 3, 3), padding='same')(tower_1)
                        tower_1 = LeakyReLU()(tower_1) #tower_1 = Activation('relu')(tower_1)
                        tower_1 = BatchNormalization()(tower_1)

                        tower_2 = Conv3D(filters, (1, 1, 1), padding='same')(layer)
                        tower_2 = Conv3D(filters, (3, 3, 3), padding='same')(tower_2)
                        tower_2 = Conv3D(filters, (3, 3, 3), padding='same')(tower_2)
                        tower_2 = LeakyReLU()(tower_2) #tower_2 = Activation('relu')(tower_2)
                        tower_2 = BatchNormalization()(tower_2)

                        tower_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1),
                                               padding='same')(layer)
                        tower_3 = Conv3D(filters, (1, 1, 1), padding='same')(tower_3)
                        tower_3 = LeakyReLU()(tower_3) #tower_3 = Activation('relu')(tower_3)
                        tower_3 = BatchNormalization()(tower_3)

                        layer = Concatenate(axis=4)([tower_0, tower_1, tower_2, tower_3])

                        layer = MaxPooling3D((2, 2, 2))(layer)

                        # layer = BatchNormalization()(layer)
                        filters = filters * 2

                elif self.input_type[iname] == 'image':

                    layer = Conv2D(64, kernel_size=(5, 5))(inputLayer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    #layer = MaxPooling2D(pool_size=(2, 2))(layer)

                    """ 
                    Core Stream 
                    """
                    # Calculate maximum number of inception blocks allowed by data size
                    (H0, W0, _) = layer.get_shape().as_list()[1:]

                    ninceptions_max = np.minimum(int(np.ceil(np.log2(H0))),
                                                 int(np.ceil(np.log2(W0))))
                    H = np.array([pool_outshape(H0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    W = np.array([pool_outshape(W0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])
                    # print(ninceptions_max,H0,W0,H,W)
                    ninceptions_max = np.maximum(0,
                                                 np.minimum(np.max(np.where(H > 1)[0]) if np.where(H > 1)[0] != [] else -1,
                                                            np.max(np.where(W > 1)[0]) if np.where(W > 1)[
                                                                                              0] != [] else -1) + 1
                                                 )

                    filters = 32
                    mz = 0
                    for icp in range(ninceptions_max):

                        tower_0 = Conv2D(filters, (1, 1), padding='same')(layer)
                        tower_0 = LeakyReLU()(tower_0) #tower_0 = Activation('relu')(tower_0)
                        tower_0 = BatchNormalization()(tower_0)

                        tower_1 = Conv2D(filters, (1, 1), padding='same', activation='relu')(layer)
                        tower_1 = Conv2D(filters, (3, 3), padding='same')(tower_1)
                        tower_1 = LeakyReLU()(tower_1) #tower_1 = Activation('relu')(tower_1)
                        tower_1 = BatchNormalization()(tower_1)

                        tower_2 = Conv2D(filters, (1, 1), padding='same', activation='relu')(layer)
                        tower_2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(tower_2)
                        tower_2 = Conv2D(filters, (3, 3), padding='same')(tower_2)
                        tower_2 = LeakyReLU()(tower_2) #tower_2 = Activation('relu')(tower_2)
                        tower_2 = BatchNormalization()(tower_2)

                        tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer)
                        tower_3 = Conv2D(filters, (1, 1), padding='same')(tower_3) # (tower_2)
                        tower_3 = LeakyReLU()(tower_3) #tower_3 = Activation('relu')(tower_3)
                        tower_3 = BatchNormalization()(tower_3)

                        layer = Concatenate(axis=3)([tower_0, tower_1, tower_2, tower_3])
                        ##################################################################
                        ##################################################################
                        ##################################################################
                        layer = MaxPooling2D((2, 2))(layer) # Comment Clecio/Luciana
                        ##################################################################
                        ##################################################################
                        ##################################################################
                        # layer = BatchNormalization()(layer)
                        filters = filters * 2
                        mz = mz + 1

                elif self.input_type[iname] == 'log':

                    # Calculate maximum number of inception blocks allowed by data size
                    (H0, W0) = inputLayer.get_shape().as_list()[1:]

                    ninceptions_max = int(np.ceil(np.log2(H0)))

                    H = np.array([pool_outshape(H0, 2, 2, times=t) \
                                  for t in np.arange(1, ninceptions_max)])

                    # print(ninceptions_max,H0,W0,H,W)
                    ninceptions_max = np.maximum(0,
                                                 np.max(np.where(H > 1)[0]) if np.where(H > 1)[0] != [] else -1
                                                 )

                    filters = 32
                    for _ in range(ninceptions_max):
                        tower_0 = Conv1D(filters, (1), padding='same')(inputLayer)
                        tower_0 = LeakyReLU()(tower_0) #tower_0 = Activation('relu')(tower_0)
                        tower_0 = BatchNormalization()(tower_0)

                        tower_1 = Conv1D(filters, (1), padding='same')(inputLayer)
                        tower_1 = Conv1D(filters, (3), padding='same')(tower_1)
                        tower_1 = LeakyReLU()(tower_1) #tower_1 = Activation('relu')(tower_1)
                        tower_1 = BatchNormalization()(tower_1)

                        tower_2 = Conv1D(filters, (1), padding='same')(inputLayer)
                        tower_2 = Conv1D(filters, (3), padding='same')(tower_2)
                        tower_2 = Conv1D(filters, (3), padding='same')(tower_2)
                        tower_2 = LeakyReLU()(tower_2) #tower_2 = Activation('relu')(tower_2)
                        tower_2 = BatchNormalization()(tower_2)

                        tower_3 = MaxPooling1D(3, strides=1, padding='same')(inputLayer)
                        tower_3 = Conv1D(filters, (1), padding='same')(tower_3)
                        tower_3 = LeakyReLU()(tower_3) #tower_3 = Activation('relu')(tower_3)
                        tower_3 = BatchNormalization()(tower_3)

                        layer = Concatenate(axis=2)([tower_0, tower_1, tower_2, tower_3])

                        layer = MaxPooling1D(2)(layer)

                        # layer = BatchNormalization()(layer)
                        filters = filters * 2

                # Flatten activations
                layer = Flatten()(layer)

                # Add stream to input streams
                inputStreams.append(layer)

            """ Mid stream """
            mid_stream = layer
            if len(self.input_shape) > 1:
                # add concatenate layer
                mid_stream = Concatenate(axis=1)(inputStreams)

            """ 
            Output Stream 
            """
            outputs = output_dense_tree(mid_stream,
                                        self.output_shape,
                                        self.last_activations,
                                        heteroscedastic=self.heteroscedastic,
                                        units=self.out_units)


        return inputs, outputs, mid_stream

""" Xception NET """
class XceptionNet(object):
    def __init__(self, input_shape, output_shape,
                 last_activations = None, heteroscedastic = False,
                 name = 'xception', out_units = [512, 256, 128, 64], **kwargs):
        
        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
                
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        self.input_type = x_type
        self.out_units = out_units
        
    """ Build dummy network """
    def build(self, **kwargs):
        
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
            
                layer = Conv3D(32, (3,3,3), strides=(2,2,2), 
                               use_bias=False)(inputLayer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                layer = Conv2D(64, (3,3,3), use_bias=False)(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                """
                Core Stream
                """
                
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,D0,_) = layer.get_shape().as_list()[1:]
                
                ninceptions_max = np.minimum(int(np.ceil(np.log2(H0)/np.log2(3))),
                                             int(np.ceil(np.log2(W0)/np.log2(3))),
                                             int(np.ceil(np.log2(D0)/np.log2(3))))
                H = np.array([pool_outshape(H0,3,2,times=t) \
                              for t in np.arange(1,ninceptions_max)])
                W = np.array([pool_outshape(W0,3,2,times=t) \
                              for t in np.arange(1,ninceptions_max)])
                D = np.array([pool_outshape(D0,3,2,times=t) \
                              for t in np.arange(1,ninceptions_max)])
                #print(ninceptions_max,H0,W0,H,W)
                ninceptions_max = np.maximum(0,
                                             np.minimum(np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else 0,
                                                        np.max(np.where(W > 1)[0]) if np.where(W>1)[0] != [] else 0,
                                                        np.max(np.where(D > 1)[0]) if np.where(D>1)[0] != [] else 0) + 1
                                             )
                
                # starting core
                filters = 128
                for i in range(ninceptions_max):
                    residual = Conv3D(filters, (1,1,1), strides=(2,2,2), \
                                      padding='same', use_bias=False)(layer)
                    residual = BatchNormalization()(residual)
                    
                    if (i != 0):
                        layer = Activation('relu')(layer)
                    
                    ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                    layer = Conv3D(filters, (3,3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    
                    ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                    layer = Conv3D(filters, (3,3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = MaxPooling3D((3,3,3), strides=(2,2,2), 
                                         padding='same')(layer)
                    
                    layer = Add()([layer, residual])
                    
                    filters *= 2
                
                filters //= 2
                # middle core
                for _ in range(8):
                    residual = layer
                    
                    layer = Activation('relu')(layer)
                    ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                    layer = Conv3D(filters, (3,3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                    layer = Conv3D(filters, (3,3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                    layer = Conv3D(filters, (3,3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = Add()([layer, residual])
                
                # finishing flow
                ### THIS IS NOT SUPPOSED TO BE A SEPARABLECNV3D
                residual = Conv3D(1024, (1,1,1), strides=(2,2,2), 
                                  padding='same', use_bias=False)(layer)
                residual = BatchNormalization()(residual)
                
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                layer = Conv3D(728, (3,3,3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                layer = Conv3D(1024, (3,3,3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                
                layer = MaxPooling3D((3,3,3), strides=(2,2,2), 
                                     padding='same')(layer)
                
                layer = Add()([layer, residual])
                
                # final blocks
                ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                layer = Conv3D(1536, (3, 3, 3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                ### WARNING!! THIS SHOULD BE A SEPARABLECONV3D
                layer = Conv3D(2048, (3, 3, 3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = GlobalAveragePooling3D()(layer)
                
            elif self.input_type[iname] == 'image':
                
                layer = Conv2D(32, (3,3), strides=(2,2), 
                               use_bias=False)(inputLayer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                layer = Conv2D(64, (3,3), use_bias=False)(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                """
                Core Stream
                """
                
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,_) = layer.get_shape().as_list()[1:]
                
                ninceptions_max = np.minimum(int(np.ceil(np.log2(H0)/np.log2(3))),
                                             int(np.ceil(np.log2(W0)/np.log2(3))))
                H = np.array([pool_outshape(H0,3,2,times=t) \
                              for t in np.arange(1,ninceptions_max)])
                W = np.array([pool_outshape(W0,3,2,times=t) \
                              for t in np.arange(1,ninceptions_max)])
                #print(ninceptions_max,H0,W0,H,W)
                ninceptions_max = np.maximum(0,
                                             np.minimum(np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else 0,
                                                        np.max(np.where(W > 1)[0]) if np.where(W>1)[0] != [] else 0) + 1
                                             )
                
                # starting core
                filters = 128
                for i in range(ninceptions_max):
                    residual = Conv2D(filters, (1,1), strides=(2,2), \
                                      padding='same', use_bias=False)(layer)
                    residual = BatchNormalization()(residual)
                    
                    if (i != 0):
                        layer = Activation('relu')(layer)
                    
                    layer = SeparableConv2D(filters, (3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    
                    
                    layer = SeparableConv2D(filters, (3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = MaxPooling2D((3,3), strides=(2,2), 
                                         padding='same')(layer)
                    
                    layer = Add()([layer, residual])
                    
                    filters *= 2
                
                filters //= 2
                # middle core
                for _ in range(8):
                    residual = layer
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = SeparableConv2D(filters, (3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = SeparableConv2D(filters, (3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = SeparableConv2D(filters, (3,3), padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = Add()([layer, residual])
                
                # finishing flow
                residual = Conv2D(1024, (1,1), strides=(2,2), 
                                  padding='same', use_bias=False)(layer)
                residual = BatchNormalization()(residual)
                
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = SeparableConv2D(728, (3,3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = SeparableConv2D(1024, (3,3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                
                layer = MaxPooling2D((3,3), strides=(2,2), 
                                     padding='same')(layer)
                
                layer = Add()([layer, residual])
                
                # final blocks
                layer = SeparableConv2D(1536, (3, 3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = SeparableConv2D(2048, (3, 3), padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = GlobalAveragePooling2D()(layer)
                
            elif self.input_type[iname] == 'log':
                
                layer = Conv1D(32, 3, strides = 2, 
                                   use_bias=False)(inputLayer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                layer = Conv1D(64, 3, use_bias=False)(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                """
                Core Stream
                """
                
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,) = layer.get_shape().as_list()[1:]
                
                ninceptions_max = int(np.ceil(np.log2(H0)/np.log2(3)))
                H = np.array([pool_outshape(H0,3,2,times=t) \
                              for t in np.arange(1,ninceptions_max)])
                #print(ninceptions_max,H0,W0,H,W)
                ninceptions_max = np.maximum(0,
                                             (np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else 0) + 1
                                             )
                
                # starting core
                filters = 128
                for i in range(ninceptions_max):
                    residual = Conv1D(filters, 1, strides = 2, \
                                      padding='same', use_bias=False)(layer)
                    residual = BatchNormalization()(residual)
                    
                    if (i != 0):
                        layer = Activation('relu')(layer)
                    
                    layer = SeparableConv1D(filters, 3, padding='same', 
                                            use_bias=False)(layer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    
                    
                    layer = SeparableConv1D(filters, 3, padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = MaxPooling1D(3, strides=2, 
                                         padding='same')(layer)
                    
                    layer = Add()([layer, residual])
                    
                    filters *= 2
                
                filters //= 2
                # middle core
                for _ in range(8):
                    residual = layer
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = SeparableConv1D(filters, 3, padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = SeparableConv1D(filters, 3, padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = SeparableConv1D(filters, 3, padding='same', 
                                            use_bias=False)(layer)
                    layer = BatchNormalization()(layer)
                    
                    layer = Add()([layer, residual])
                
                # finishing flow
                residual = Conv1D(1024, 1, strides = 2, 
                                  padding='same', use_bias=False)(layer)
                residual = BatchNormalization()(residual)
                
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = SeparableConv1D(728, 3, padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = SeparableConv1D(1024, 3, padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                
                layer = MaxPooling1D(3, strides = 2, 
                                     padding='same')(layer)
                
                layer = Add()([layer, residual])
                
                # final blocks
                layer = SeparableConv1D(1536, 3, padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = SeparableConv1D(2048, 3, padding='same', 
                                        use_bias=False)(layer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                layer = GlobalAveragePooling1D()(layer)
                
            # Add stream to input streams
            inputStreams.append(layer)
            
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
            
        """ 
        Output Stream 
        """
        outputs = output_dense_tree(mid_stream, 
                                    self.output_shape, 
                                    self.last_activations,
                                    heteroscedastic = self.heteroscedastic,
                                    units = self.out_units)
        
        return inputs, outputs, mid_stream




""" ResNet NET """
class ResNet(object):
    def __init__(self, input_shape, output_shape, repetitions = [2, 2],
                 last_activations = None, heteroscedastic = False,
                 name = 'resnet', out_units = [512, 256, 128, 64], **kwargs):
        
        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.repetitions = repetitions
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        self.out_units = out_units
        
        x_type = {xn: 'volume' if len(input_shape[xn][1:]) == 4 else \
                      'image' if len(input_shape[xn][1:]) == 3 else'log'\
                  for xn in input_shape}
        self.input_type = x_type
        
    """ Build resnet network """
    def build(self, **kwargs):
        
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
                
                layer = Conv3D(64,(3,3,3),strides=(2,2,2))(inputLayer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #ayer = Activation('relu')(layer)
                                
                """ 
                Core Stream 
                """
                layer = MaxPooling3D(pool_size = (3,3,3), strides = (2,2,2), 
                                     padding = "same")(layer)
                filters = 64
                for i,r in enumerate(self.repetitions):
                    for ii in range(r):
                        project_shortcut = True if ii == 0 else False
                        layer = residual_block(layer, filters, filters*2,
                                               bayesian = False,
                                               _project_shortcut = project_shortcut)
                    filters *= 2
                    
                # Last activation
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2), 
                                     padding="same")(layer)
            
            elif self.input_type[iname] == 'image':
                
                layer = Conv2D(64,(3,3),strides=(2,2), kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(inputLayer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                
                #inputStreams.append(layer)
                
                """ 
                Core Stream 
                """
                layer = MaxPooling2D(pool_size=(3,3), strides=(2,2), 
                                     padding="same")(layer)
                filters = 64
                for i,r in enumerate(self.repetitions):
                    for ii in range(r):
                        project_shortcut = True if ii == 0 else False
                        # layer = residual_block(layer, filters, filters*2,
                        #                        bayesian = False,
                        #                        _project_shortcut = project_shortcut)
                        layer = residual_block(layer, filters, filters*2,
                                               _project_shortcut = project_shortcut) # change (error bayesian not defined)
                    filters *= 2
                    
                # Last activation
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling2D(pool_size=(3,3), strides=(2,2), 
                                     padding="same")(layer)
                
            elif self.input_type[iname] == 'log':
                
                layer = Conv1D(64, 3, strides = 2)(inputLayer)
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                                    
                """ 
                Core Stream 
                """
                layer = MaxPooling1D(pool_size = 3, strides = 2, 
                                     padding = "same")(layer)
                filters = 64
                for i,r in enumerate(self.repetitions):
                    for ii in range(r):
                        project_shortcut = True if ii == 0 else False
                        layer = residual_block(layer, filters, filters*2,
                                               _project_shortcut = project_shortcut)
                    filters *= 2
                    
                # Last activation
                layer = BatchNormalization()(layer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = MaxPooling1D(pool_size = 3, strides = 2, 
                                     padding = "same")(layer)
                
            
            # Flatten activations
            layer = Flatten()(layer)

            # Add stream to input streams
            inputStreams.append(layer)
            
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
            
        """ 
        Output Stream 
        """
        outputs = output_dense_tree(mid_stream, 
                                    self.output_shape, 
                                    self.last_activations,
                                    heteroscedastic = self.heteroscedastic,
                                    units = self.out_units)
        
        return inputs, outputs, mid_stream


""" ResNeXt NET """
class ResNeXt(object):
    def __init__(self, input_shape, output_shape, cardinality = 32,
                 last_activations = None, heteroscedastic = False,
                 name = 'resnext', out_units = [512, 256, 128, 64], **kwargs):
        
        """ Set Parameters """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.cardinality = cardinality
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        self.out_units = out_units
        
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        self.input_type = x_type
        
    """ Build resnext network """
    def build(self, **kwargs):
        
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
                
                #conv1  + relu + bn
                layer = Conv2D(64,kernel_size=(7,7,7),strides=(2,2,2),
                               padding='same')(inputLayer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                """
                Core Stream
                """
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,D0,_) = layer.get_shape().as_list()[1:]
                
                nblocks_max = np.minimum(int(np.ceil(np.log2(H0)/np.log2(3))),
                                         int(np.ceil(np.log2(W0)/np.log2(3))),
                                         int(np.ceil(np.log2(D0)/np.log2(3))))
                
                H = np.array([pool_outshape(H0,3,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                W = np.array([pool_outshape(W0,3,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                D = np.array([pool_outshape(D0,3,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
    
                #print(ninceptions_max,H0,W0,H,W)
                nblocks_max = np.maximum(0,
                                         np.minimum(np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else -1,
                                                    np.max(np.where(W > 1)[0]) if np.where(W>1)[0] != [] else -1,
                                                    np.max(np.where(D > 1)[0]) if np.where(D>1)[0] != [] else -1) + 1
                                         )
                
                filters = 96
                for _ in range(nblocks_max):
                    layer = MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2),
                                         padding='same')(layer)
                                        
                    layer = residual_block(layer, filters, filters*2, 
                                           cardinality = self.cardinality,
                                           _project_shortcut = True)
                    
                    layer = BatchNormalization()(layer)
                    
                    filters *= 2
            
            elif self.input_type[iname] == 'image':
                
                #conv1  + relu + bn
                layer = Conv2D(64,kernel_size=(7,7),strides=(2,2),
                               padding='same')(inputLayer)
                layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                layer = BatchNormalization()(layer)
                
                """
                Core Stream
                """
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,_) = layer.get_shape().as_list()[1:]
                
                nblocks_max = np.minimum(int(np.ceil(np.log2(H0)/np.log2(3))),
                                             int(np.ceil(np.log2(W0)/np.log2(3))))
                H = np.array([pool_outshape(H0,3,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                W = np.array([pool_outshape(W0,3,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                #print(ninceptions_max,H0,W0,H,W)
                nblocks_max = np.maximum(0,
                                             np.minimum(np.max(np.where(H > 1)[0]) if np.where(H>1)[0] != [] else -1,
                                                        np.max(np.where(W > 1)[0]) if np.where(W>1)[0] != [] else -1) + 1
                                             )
                
                filters = 96
                for _ in range(nblocks_max):
                    layer = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                                         padding='same')(layer)
                    
                    layer = residual_block(layer, filters, filters*2, 
                                           cardinality = self.cardinality,
                                           _project_shortcut = True)
                    
                    layer = BatchNormalization()(layer)
                    
                    filters *= 2
                
            elif self.input_type[iname] == 'log':
                
                if len(self.input_shape[iname][1:]) > 1:
                    #conv1  + relu + bn
                    layer = Conv1D(64, kernel_size=(7),strides=(2),
                                   padding='same')(inputLayer)
                    layer = LeakyReLU()(layer) #layer = Activation('relu')(layer)
                    layer = BatchNormalization()(layer)
                    
                    """ Core Stream """
                    # Calculate maximum number of inception blocks allowed by data size
                    (H0,W0) = layer.get_shape().as_list()[1:]
                    
                    nblocks_max = int(np.ceil(np.log2(H0)/np.log2(3)))
                    
                    H = np.array([pool_outshape(H0,3,2,times=t) \
                                  for t in np.arange(1,nblocks_max)])
                    
                    #print(ninceptions_max,H0,W0,H,W)
                    nblocks_max = np.maximum(0,
                                                 np.max(np.where(H > 1)[0]) \
                                                     if np.where(H>1)[0] != [] else -1
                                                 )
                    
                    filters = 96
                    for _ in range(nblocks_max):
                        layer = MaxPooling1D(pool_size = 3, strides = 2,
                                             padding = 'same')(layer)
                        
                        layer = residual_block(layer, filters, filters*2, 
                                               cardinality = self.cardinality,
                                               _project_shortcut = True)
                        
                        layer = BatchNormalization()(layer)
                        
                        filters *= 2
                    
                    # Flatten activations
                    layer = Flatten()(layer)
                else:
                    layer = inputLayer
                    units = [64, 128, 256]
                    for u in units:
                        layer = Dense(u)(layer)
                        layer = BatchNormalization()(layer)
                        layer = Activation('relu')(layer)
                    
            
            

            # Add stream to input streams
            inputStreams.append(layer)
            
        """ Mid stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
            
        """ 
        Output Stream 
        """
        outputs = output_dense_tree(mid_stream, 
                                    self.output_shape, 
                                    self.last_activations,
                                    heteroscedastic = self.heteroscedastic,
                                    units = self.out_units)
        
        return inputs, outputs, mid_stream


""" URESNET """
class UResNet(object):
    def __init__(self, input_shape, output_shape, 
                 num_inner_encoders = 3, 
                 encoder_units = 64,
                 dense_units = 1024,
                 dropout_rate = 0.5,
                 last_activations = None,
                 heteroscedastic = False,
                 name = 'uresnet',
                 out_units = [512, 256, 128, 64],
                 **kwargs):
        
        """Set Parameters"""
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_inner_encoders = num_inner_encoders
        self.encoder_units = encoder_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.name = name
        if last_activations is None:
            last_activations = {xn: None for xn in self.input_shape}
        self.last_activations = last_activations
        self.heteroscedastic = heteroscedastic
        self.out_units = out_units
        x_type = {xn: 'image' if len(input_shape[xn][1:]) > 2 else 'log'\
                  for xn in input_shape}
        
        self.input_type = x_type
        
    """ Build UResNet object """
    def build(self, **kwargs):
        
        """ 
        Input Stream 
        """
        inputs = []
        inputStreams = []
        for iname in self.input_shape:
            inputLayer = Input(shape = self.input_shape[iname][1:],
                               name = iname)
            # add input layer to inputs
            inputs.append(inputLayer)
            
            if self.input_type[iname] == 'volume':
            
                # Save into a variable all layers that will be linked between big blocks
                outter_links = [inputLayer]
                encoder_layers = []
                layer = inputLayer
                
                """Input stream"""
                filters = self.encoder_units
                # First layer is 2stride conv2d with leaky and bnorm, for better
                # results
                layer = self.conv3d_layer(layer, filters, 
                                        kernel_size = (3,3,3), 
                                        strides = (2,2,2))
                
                layer = LeakyReLU(0.2)(layer)
                layer = BatchNormalization(momentum = 0.99, 
                                           scale = True,
                                           center = True)(layer)
                # Add block to outter_links
                outter_links.append(layer)
                
                """Big encoding blocks loop"""
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,D0,_) = layer.get_shape().as_list()[1:]
                
                nblocks_max = np.minimum(int(np.ceil(np.log2(H0))),
                                         int(np.ceil(np.log2(W0))),
                                         int(np.ceil(np.log2(D0))))
                
                H = np.array([pool_outshape(H0,1,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                W = np.array([pool_outshape(W0,1,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                D = np.array([pool_outshape(D0,1,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
    
                #print(ninceptions_max,H0,W0,H,W)
                nblocks_max = np.maximum(0,
                                         np.minimum(np.max(np.where(H > 1)[0]) \
                                                        if np.where(H>1)[0] != [] else -1,
                                                    np.max(np.where(W > 1)[0]) \
                                                        if np.where(W>1)[0] != [] else -1,
                                                    np.max(np.where(D > 1)[0]) \
                                                        if np.where(D>1)[0] != [] else -1) + 1
                                         )
        
                
                for i in range(nblocks_max):
                    """Inner Encoders blocks"""
                    inner_links = [layer]
                    """Loop through all encoders"""
                    for j in range(self.num_inner_encoders):
                        connected_layers = self.remove_duplicates(inner_links + [outter_links[-2]])
                        # Add encoder block
                        layer = self.encoder3d_block(layer, filters, connected_layers, 
                                                   dropout_rate = self.dropout_rate,
                                                   dim_reduce = False,
                                                   name = 'encoder_layer_{}_{}'.format(i,j))
                        
                        # Add block output to inner_links
                        inner_links += [layer]
                    outter_links.append(layer)
                    """Final encoder (inside block) to reduce dimensionality"""
                    # Add encoder block
                    connected_layers = self.remove_duplicates(inner_links + [outter_links[-2]])
                    
                    layer = self.encoder3d_block(layer, filters, 
                                                 connected_layers, 
                                                 dropout_rate = self.dropout_rate,
                                                 dim_reduce = True,
                                                 name = 'encoder_layer_{}'.format(i))
                    # Add layer activation to outter_links
                    outter_links.append(layer)
                    encoder_layers.append(layer)
                    
                    if (i+1)%2 == 0:
                        filters *= 2
                
            elif self.input_type[iname] == 'image':
                # Save into a variable all layers that will be linked between big blocks
                outter_links = [inputLayer]
                encoder_layers = []
                layer = inputLayer
                
                """Input stream"""
                filters = self.encoder_units
                # First layer is 2stride conv2d with leaky and bnorm, for better
                # results
                layer = self.conv2d_layer(layer, filters, 
                                        kernel_size = (3,3), 
                                        strides = (2,2))
                
                layer = LeakyReLU(0.2)(layer)
                layer = BatchNormalization(momentum = 0.99, 
                                           scale = True,
                                           center = True)(layer)
                # Add block to outter_links
                outter_links.append(layer)
                
                """Big encoding blocks loop"""
                # Calculate maximum number of inception blocks allowed by data size
                (H0,W0,_) = layer.get_shape().as_list()[1:]
                
                nblocks_max = np.minimum(int(np.ceil(np.log2(H0))),
                                             int(np.ceil(np.log2(W0))))
                H = np.array([pool_outshape(H0,1,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                W = np.array([pool_outshape(W0,1,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                #print(ninceptions_max,H0,W0,H,W)
                nblocks_max = np.maximum(0,
                                         np.minimum(np.max(np.where(H > 1)[0]) \
                                                        if np.where(H>1)[0] != [] else -1,
                                                    np.max(np.where(W > 1)[0]) \
                                                        if np.where(W>1)[0] != [] else -1) + 1
                                         )
        
                
                for i in range(nblocks_max):
                    """Inner Encoders blocks"""
                    inner_links = [layer]
                    """Loop through all encoders"""
                    for j in range(self.num_inner_encoders):
                        connected_layers = self.remove_duplicates(inner_links + [outter_links[-2]])
                        # Add encoder block
                        layer = self.encoder2d_block(layer, filters, connected_layers, 
                                                   dropout_rate = self.dropout_rate,
                                                   dim_reduce = False,
                                                   name = 'encoder_layer_{}_{}'.format(i,j))
                        
                        # Add block output to inner_links
                        inner_links += [layer]
                    outter_links.append(layer)
                    """Final encoder (inside block) to reduce dimensionality"""
                    # Add encoder block
                    connected_layers = self.remove_duplicates(inner_links + [outter_links[-2]])
                    
                    layer = self.encoder2d_block(layer, filters, 
                                                 connected_layers, 
                                                 dropout_rate = self.dropout_rate,
                                                 dim_reduce = True,
                                                 name = 'encoder_layer_{}'.format(i))
                    # Add layer activation to outter_links
                    outter_links.append(layer)
                    encoder_layers.append(layer)
                    
                    if (i+1)%2 == 0:
                        filters *= 2
            
            else:
                """ This is a log """
                # Save into a variable all layers that will be linked between big blocks
                outter_links = [inputLayer]
                encoder_layers = []
                layer = inputLayer
                
                """Input stream"""
                filters = self.encoder_units
                # First layer is 2stride conv2d with leaky and bnorm, for better
                # results
                layer = self.conv1d_layer(layer, filters, 
                                        kernel_size = 3, 
                                        strides = 2)
                
                layer = LeakyReLU(0.2)(layer)
                layer = BatchNormalization(momentum = 0.99, 
                                           scale = True,
                                           center = True)(layer)
                # Add block to outter_links
                outter_links.append(layer)
                
                """Big encoding blocks loop"""
                # Calculate maximum number of inception blocks allowed by data size
                (H0,_) = layer.get_shape().as_list()[1:]
                
                nblocks_max = int(np.ceil(np.log2(H0)))
                H = np.array([pool_outshape(H0,1,2,times=t) \
                              for t in np.arange(1,nblocks_max)])
                #print(ninceptions_max,H0,W0,H,W)
                nblocks_max = np.maximum(0,
                                         (np.max(np.where(H > 1)[0]) \
                                                        if np.where(H>1)[0] != [] else -1) + 1
                                         )
        
                
                for i in range(nblocks_max):
                    """Inner Encoders blocks"""
                    inner_links = [layer]
                    """Loop through all encoders"""
                    for j in range(self.num_inner_encoders):
                        connected_layers = self.remove_duplicates(inner_links + [outter_links[-2]])
                        # Add encoder block
                        layer = self.encoder1d_block(layer, filters, connected_layers, 
                                                   dropout_rate = self.dropout_rate,
                                                   dim_reduce = False,
                                                   name = 'encoder_layer_{}_{}'.format(i,j))
                        
                        # Add block output to inner_links
                        inner_links += [layer]
                    outter_links.append(layer)
                    """Final encoder (inside block) to reduce dimensionality"""
                    # Add encoder block
                    connected_layers = self.remove_duplicates(inner_links + [outter_links[-2]])
                    
                    layer = self.encoder1d_block(layer, filters, 
                                                 connected_layers, 
                                                 dropout_rate = self.dropout_rate,
                                                 dim_reduce = True,
                                                 name = 'encoder_layer_{}'.format(i))
                    # Add layer activation to outter_links
                    outter_links.append(layer)
                    encoder_layers.append(layer)
                    
                    if (i+1)%2 == 0:
                        filters *= 2
            
            
            """Concatenation of flattn'd encoded features"""
            dense_units = self.dense_units
            flattened = []
            for i,(ely) in enumerate(encoder_layers):
                flat_tmp = Flatten()(ely)
                flat_tmp = Dense(dense_units)(flat_tmp)
                flat_tmp = LeakyReLU(0.2)(flat_tmp)
                flattened.append(flat_tmp)
                
            """Concatenation"""
            layer = Concatenate(axis=1)(flattened)
            
            inputStreams.append(layer)
            
            
        """ Middle Stream """
        mid_stream = layer
        if len(self.input_shape) > 1:
            # add concatenate layer
            mid_stream = Concatenate(axis=1)(inputStreams)
        
        """ 
        Output Stream 
        """
        outputs = output_dense_tree(mid_stream, 
                                    self.output_shape, 
                                    self.last_activations,
                                    heteroscedastic = self.heteroscedastic,
                                    units = self.out_units)
        
        
        return inputs, outputs, mid_stream
    
    """
    Upscale image (interpolation)
    """
    def upscale1d(self, x, h_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        try:
            out = Lambda(lambda image: K.tf.image.resize_images(image, (h_size, 1)))(x)
        except :
            # if you have older version of tensorflow
            out = Lambda(lambda image: K.tf.image.resize_images(image, h_size, 1))(x)
        return out
        
    
    """
    Upscale image (interpolation)
    """
    def upscale2d(self, x, h_size, w_size):
        """
        Upscales an image using nearest neighbour
        :param x: Input image
        :param h_size: Image height size
        :param w_size: Image width size
        :return: Upscaled image
        """
        try:
            out = Lambda(lambda image: K.tf.image.resize_images(image, (h_size, w_size)))(x)
        except :
            # if you have older version of tensorflow
            out = Lambda(lambda image: K.tf.image.resize_images(image, h_size, w_size))(x)
        return out
        
    
    """
    Upscale volume (interpolation)
    """
    def upscale3d(self, x, h_size, w_size, d_size):
        """
        Upscales an volume using nearest neighbour
        :param x: Input image
        :param h_size: volume height size
        :param w_size: volume width size
        :param d_size: volume depth size
        :return: Upscaled volume
        """
        
        def resize_by_axis(image, dim_1, dim_2, ax, is_grayscale):

            resized_list = []
        
        
            if is_grayscale:
                unstack_img_depth_list = [K.tf.expand_dims(x,2) \
                                          for x in K.tf.unstack(image, axis = ax)]
                for i in unstack_img_depth_list:
                    resized_list.append(K.tf.image.resize_images(i, [dim_1, dim_2],method=0))
                stack_img = K.tf.squeeze(K.tf.stack(resized_list, axis=ax))
                print(stack_img.get_shape())
        
            else:
                unstack_img_depth_list = K.tf.unstack(image, axis = ax)
                for i in unstack_img_depth_list:
                    resized_list.append(K.tf.image.resize_images(i, [dim_1, dim_2],method=0))
                stack_img = K.tf.stack(resized_list, axis=ax)
        
            return stack_img
        
        def resize_volume(volume, h_size, w_size, d_size):
            resized_along_depth = resize_by_axis(x,h_size,w_size,2, True)
            resized_along_width = resize_by_axis(resized_along_depth,h_size,d_size,1,True)
            
            return resized_along_width
        
        out = Lambda(lambda volume: resize_volume(volume, (h_size, w_size, d_size)))(x)
        
        return out
    
    
    """
    Convolutional layer structure (conv and deconv)
    """
    def conv1d_layer(self, inputs, num_filters, 
                   kernel_size = 3, strides = 1, 
                   activation = None, transpose = False, 
                   padding = 'same', w_size = None, h_size = None):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :return: Convolution features
        """
        if transpose:
            outputs = self.upscale1d(inputs, h_size = h_size)
            
            outputs = Conv1DTranspose(num_filters, kernel_size,
                                      strides = strides,
                                      padding = padding,
                                      activation = activation)(outputs)
            
            """
            outputs = tf.layers.conv2d_transpose(outputs, 
                                                 num_filters, 
                                                 kernel_size,
                                                 strides = strides,
                                                 padding = padding, 
                                                 activation = activation)
            """
        elif not transpose:
            outputs = Conv1D(num_filters, kernel_size, 
                             strides = strides,
                             padding = padding, 
                             activation = activation)(inputs)

        return outputs
    
    """
    Convolutional layer structure (conv and deconv)
    """
    def conv2d_layer(self, inputs, num_filters, 
                   kernel_size = (3,3), strides = (1, 1), 
                   activation = None, transpose = False, 
                   padding = 'same', w_size = None, h_size = None):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :return: Convolution features
        """
        if transpose:
            outputs = self.upscale2d(inputs, 
                                   h_size = h_size, 
                                   w_size = w_size)
            
            outputs = Conv2DTranspose(num_filters, kernel_size,
                                      strides = strides,
                                      padding = padding,
                                      activation = activation)(outputs)
            
            """
            outputs = tf.layers.conv2d_transpose(outputs, 
                                                 num_filters, 
                                                 kernel_size,
                                                 strides = strides,
                                                 padding = padding, 
                                                 activation = activation)
            """
        elif not transpose:
            outputs = Conv2D(num_filters, kernel_size, 
                             strides = strides,
                             padding = padding, 
                             activation = activation)(inputs)

        return outputs
    
    
    """
    Convolutional layer structure (conv and deconv)
    """
    def conv3d_layer(self, inputs, num_filters, 
                   kernel_size = (3,3,3), strides = (1, 1, 1), 
                   activation = None, transpose = False, 
                   padding = 'same', w_size = None, h_size = None, d_size = None):
        """
        Add a convolutional layer to the network.
        :param inputs: Inputs to the conv layer.
        :param num_filters: Num of filters for conv layer.
        :param filter_size: Size of filter.
        :param strides: Stride size.
        :param activation: Conv layer activation.
        :param transpose: Whether to apply upscale before convolution.
        :param w_size: Used only for upscale, w_size to scale to.
        :param h_size: Used only for upscale, h_size to scale to.
        :param d_size: Used only for upscale, d_size to scale to.
        :return: Convolution features
        """
        if transpose:
            outputs = self.upscale3d(inputs, 
                                   h_size = h_size, 
                                   w_size = w_size,
                                   d_size = d_size)
            
            outputs = Conv3DTranspose(num_filters, kernel_size,
                                      strides = strides,
                                      padding = padding,
                                      activation = activation)(outputs)
            
        elif not transpose:
            outputs = Conv3D(num_filters, kernel_size, 
                             strides = strides,
                             padding = padding, 
                             activation = activation)(inputs)

        return outputs
    
    
    """
    Encoder structure
    """
    def encoder1d_block(self, inputs, num_filters, connected_layers, 
                      dropout_rate = 0.5, dim_reduce = False, name = None):
        
        # Make sure all connected_activations will have the same size
        # if not, put a conv layer in the middle
        [b1, h1, d1] = inputs.get_shape().as_list()

        for li, ly in enumerate(connected_layers):
            [b0, h0, d0] = ly.get_shape().as_list()
            if h0 > h1:
                skip_layer = self.conv1d_layer(ly, d0, 
                                               kernel_size = 3, 
                                               strides = 2)
                # replace layer
                connected_layers[li] = skip_layer
        
        # Place input on connected_layers
        connected_layers += [inputs]
        connected_layers = self.remove_duplicates(connected_layers)
        
        # Now concatenate connected_activations
        output = Concatenate(axis=2)(connected_layers)
        
        # Final convolutional block
        if dim_reduce:
            # reduce dimension by 2 (stride = 2)
            output = self.conv1d_layer(output, num_filters, 
                                       kernel_size = 3,
                                       strides = 2)
            output = LeakyReLU(0.2)(output)
            output = BatchNormalization(momentum = 0.99, 
                                        scale = True,
                                        center = True)(output)
            output = Dropout(dropout_rate)(output)
        else:
            # keep dimensions
            output = self.conv1d_layer(output, num_filters, 
                                       kernel_size = 3,
                                       strides = 1)
            output = LeakyReLU(0.2)(output)
            output = BatchNormalization(momentum = 0.99, 
                                        scale = True,
                                        center = True)(output)
            
        return output
    
    
    
    """
    Encoder structure
    """
    def encoder2d_block(self, inputs, num_filters, connected_layers, 
                      dropout_rate = 0.5, dim_reduce = False, name = None):
        
        # Make sure all connected_activations will have the same size
        # if not, put a conv layer in the middle
        [b1, h1, w1, d1] = inputs.get_shape().as_list()

        for li, ly in enumerate(connected_layers):
            [b0, h0, w0, d0] = ly.get_shape().as_list()
            if h0 > h1:
                skip_layer = self.conv2d_layer(ly, d0, kernel_size = (3,3), 
                                        strides = (2,2))
                # replace layer
                connected_layers[li] = skip_layer
        
        # Place input on connected_layers
        connected_layers += [inputs]
        connected_layers = self.remove_duplicates(connected_layers)
        
        # Now concatenate connected_activations
        output = Concatenate(axis=3)(connected_layers)
        
        # Final convolutional block
        if dim_reduce:
            # reduce dimension by 2 (stride = 2)
            output = self.conv2d_layer(output, num_filters, 
                                       kernel_size = (3,3),
                                       strides = (2,2))
            output = LeakyReLU(0.2)(output)
            output = BatchNormalization(momentum = 0.99, 
                                        scale = True,
                                        center = True)(output)
            output = Dropout(dropout_rate)(output)
        else:
            # keep dimensions
            output = self.conv2d_layer(output, num_filters, 
                                       kernel_size = (3,3),
                                       strides = (1,1))
            output = LeakyReLU(0.2)(output)
            output = BatchNormalization(momentum = 0.99, 
                                        scale = True,
                                        center = True)(output)
            
        return output
    
    """
    Encoder structure
    """
    def encoder3d_block(self, inputs, num_filters, connected_layers, 
                      dropout_rate = 0.5, dim_reduce = False, name = None):
        
        # Make sure all connected_activations will have the same size
        # if not, put a conv layer in the middle
        [b1, h1, w1, k1, d1] = inputs.get_shape().as_list()

        for li, ly in enumerate(connected_layers):
            [b0, h0, w0, k0, d0] = ly.get_shape().as_list()
            if h0 > h1:
                skip_layer = self.conv3d_layer(ly, d0, kernel_size = (3,3,3), 
                                        strides = (2,2,2))
                # replace layer
                connected_layers[li] = skip_layer
        
        # Place input on connected_layers
        connected_layers += [inputs]
        connected_layers = self.remove_duplicates(connected_layers)
        
        # Now concatenate connected_activations
        output = Concatenate(axis=4)(connected_layers)
        
        # Final convolutional block
        if dim_reduce:
            # reduce dimension by 2 (stride = 2)
            output = self.conv3d_layer(output, num_filters, 
                                       kernel_size = (3,3,3),
                                       strides = (2,2,2))
            output = LeakyReLU(0.2)(output)
            output = BatchNormalization(momentum = 0.99, 
                                        scale = True,
                                        center = True)(output)
            output = Dropout(dropout_rate)(output)
        else:
            # keep dimensions
            output = self.conv3d_layer(output, num_filters, 
                                       kernel_size = (3,3,3),
                                       strides = (1,1,1))
            output = LeakyReLU(0.2)(output)
            output = BatchNormalization(momentum = 0.99, 
                                        scale = True,
                                        center = True)(output)
            
        return output
   

    """
    Remove duplicate tensors
    """
    def remove_duplicates(self, input_features):
        """
        Remove duplicate entries from layer list.
        :param input_features: A list of layers
        :return: Returns a list of unique feature tensors (i.e. no duplication).
        """
        feature_name_set = set()
        non_duplicate_feature_set = []
        for feature in input_features:
            if feature.name not in feature_name_set:
                non_duplicate_feature_set.append(feature)
            feature_name_set.add(feature.name)
        return non_duplicate_feature_set
    
