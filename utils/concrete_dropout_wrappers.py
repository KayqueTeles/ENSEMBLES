#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:06:28 2019

@author: mbvalentin
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
import numpy as np
np.random.seed(0)
try:
    import keras.backend as K
    from keras import initializers
    from keras.engine import InputSpec
    from keras.layers import Wrapper
except:
    import tensorflow.keras.backend as K
    from tensorflow.keras import initializers
    from tensorflow.keras.layers import InputSpec, Wrapper

class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer = 1e-6, dropout_regularizer = 1e-5,
                 init_min = 0.1, init_max = 0.1, is_mc_dropout = True, 
                 data_format=None, **kwargs):
        
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        
        """ Init params """
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'
        
        # Initialize concretedropout type (which we will set when the layer is
        # called)
        self.type = None

    """ Build Method """
    def build(self, input_shape = None):
        
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
            
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses
        
        # initialise p
        self.p_logit = self.layer.add_weight(name = 'p_logit',
                                            shape = (1,),
                                            initializer = initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable = True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        """ Parse layer type """
        if len(input_shape) == 5:
            """ Conv2D """            
            # Drop only the channels
            if self.data_format == 'channels_first':
                input_dim = input_shape[1] # we drop only channels
            else:
                input_dim = input_shape[4]
            
            # get weights
            if self.layer.__class__.__name__ == "Conv3D":
                weight = self.layer.kernel
            elif self.layer.__class__.__name__ == "Dense":
                weight = self.layer.kernel
            elif (self.layer.__class__.__name__ == "SeparableConv3D"):
                weight = self.layer.weights[0]
            
            """ Initialize KR and loss """
            kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
            dropout_regularizer = self.p * K.log(self.p)
            dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = K.sum(kernel_regularizer + dropout_regularizer)
            self.layer.add_loss(regularizer)
            
            # Set Dropout type
            self.type = 'conv3d'
            
        elif len(input_shape) == 4:
            """ Conv2D """            
            # Drop only the channels
            if self.data_format == 'channels_first':
                input_dim = input_shape[1] # we drop only channels
            else:
                input_dim = input_shape[3]
            
            # get weights
            if self.layer.__class__.__name__ == "Conv2D":
                weight = self.layer.kernel
            elif self.layer.__class__.__name__ == "Dense":
                weight = self.layer.kernel
            elif (self.layer.__class__.__name__ == "SeparableConv2D"):
                weight = self.layer.weights[0]
            
            """ Initialize KR and loss """
            kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
            dropout_regularizer = self.p * K.log(self.p)
            dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = K.sum(kernel_regularizer + dropout_regularizer)
            self.layer.add_loss(regularizer)
            
            # Set Dropout type
            self.type = 'conv2d'
            
        elif len(input_shape) == 3:
            """ Conv1D """
            # Drop only the channels
            if self.data_format == 'channels_first':
                input_dim = input_shape[1] # we drop only channels
            else:
                input_dim = input_shape[2]
            
            # get weights
            if self.layer.__class__.__name__ == "Conv1D":
                weight = self.layer.kernel
            elif self.layer.__class__.__name__ == "Dense":
                weight = self.layer.kernel
            elif (self.layer.__class__.__name__ == "SeparableConv1D"):
                weight = self.layer.weights[0]
            
            """ Initialize KR and loss """
            kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
            dropout_regularizer = self.p * K.log(self.p)
            dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = K.sum(kernel_regularizer + dropout_regularizer)
            self.layer.add_loss(regularizer)
            
            # Set Dropout type
            self.type = 'conv1d'
            
        elif len(input_shape) == 2:
            """ Pure shallow dense """
            input_dim = np.prod(input_shape[-1])  # we drop only last dim
            
            # get weights
            weight = self.layer.kernel
            
            """ Initialize KR and loss """
            kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
            dropout_regularizer = self.p * K.log(self.p)
            dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
            dropout_regularizer *= self.dropout_regularizer * input_dim
            regularizer = K.sum(kernel_regularizer + dropout_regularizer)
            self.layer.add_loss(regularizer)
            
            # Set Dropout type
            self.type = 'dense'
            
        else:
            raise Exception('Invalid layer shape. Dropout wrappers only works with '\
                  'tensors of 2, 3 or 4 dimensions but the shape of the '\
                  'input tensor is {}.'.format(input_shape) )

            
    """ Output shape computation """
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    """ The actual dropout method """
    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        
        if self.type == 'dense':
            eps = K.cast_to_floatx(K.epsilon())
            temp = 0.1

            unif_noise = K.random_uniform(shape=K.shape(x))
            drop_prob = (
                K.log(self.p + eps)
                - K.log(1. - self.p + eps)
                + K.log(unif_noise + eps)
                - K.log(1. - unif_noise + eps)
            )
            drop_prob = K.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob
    
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
            
        elif self.type == 'conv1d':
            eps = K.cast_to_floatx(K.epsilon())
            temp = 2. / 3.
    
            input_shape = K.shape(x)
            if self.data_format == 'channels_first':
                noise_shape = (input_shape[0], input_shape[1], 1)
            else:
                noise_shape = (input_shape[0], 1, input_shape[2])
            unif_noise = K.random_uniform(shape=noise_shape)
            
            drop_prob = (
                K.log(self.p + eps)
                - K.log(1. - self.p + eps)
                + K.log(unif_noise + eps)
                - K.log(1. - unif_noise + eps)
            )
            drop_prob = K.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob
    
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
            
        elif self.type == 'conv2d':
            eps = K.cast_to_floatx(K.epsilon())
            temp = 2. / 3.
    
            input_shape = K.shape(x)
            if self.data_format == 'channels_first':
                noise_shape = (input_shape[0], input_shape[1], 1, 1)
            else:
                noise_shape = (input_shape[0], 1, 1, input_shape[3])
            unif_noise = K.random_uniform(shape=noise_shape)
            
            drop_prob = (
                K.log(self.p + eps)
                - K.log(1. - self.p + eps)
                + K.log(unif_noise + eps)
                - K.log(1. - unif_noise + eps)
            )
            drop_prob = K.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob
    
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
        
        elif self.type == 'conv3d':
            eps = K.cast_to_floatx(K.epsilon())
            temp = 2. / 3.
    
            input_shape = K.shape(x)
            if self.data_format == 'channels_first':
                noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)
            else:
                noise_shape = (input_shape[0], 1, 1, 1, input_shape[3])
            unif_noise = K.random_uniform(shape=noise_shape)
            
            drop_prob = (
                K.log(self.p + eps)
                - K.log(1. - self.p + eps)
                + K.log(unif_noise + eps)
                - K.log(1. - unif_noise + eps)
            )
            drop_prob = K.sigmoid(drop_prob / temp)
            random_tensor = 1. - drop_prob
    
            retain_prob = 1. - self.p
            x *= random_tensor
            x /= retain_prob
        
        return x

    """ Call function (returns a dropped activation)"""
    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)