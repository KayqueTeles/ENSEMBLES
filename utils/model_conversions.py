#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:04:58 2019

@author: mbvalentin
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

""" Concrete Dropout """
from .concrete_dropout_wrappers import ConcreteDropout

""" Keras Layers """
try:
    from keras.layers import (Dense, Activation, Dropout, Flatten, Input, 
                              Reshape, Permute, RepeatVector, Lambda, 
                              ActivityRegularization, Masking, SpatialDropout1D,
                              SpatialDropout2D, SpatialDropout3D,
                              Conv1D, Conv2D, Conv3D, SeparableConv1D,
                              SeparableConv2D, Conv2DTranspose, 
                              Conv3DTranspose, Cropping1D, Cropping2D,
                              Cropping3D, UpSampling1D, UpSampling2D,
                              UpSampling3D, ZeroPadding1D, ZeroPadding2D,
                              ZeroPadding3D, MaxPooling1D, MaxPooling2D,
                              MaxPooling3D, AveragePooling1D, AveragePooling2D,
                              AveragePooling3D, GlobalMaxPooling1D, 
                              GlobalMaxPooling2D, GlobalMaxPooling3D,
                              GlobalAveragePooling1D, GlobalAveragePooling2D,
                              GlobalAveragePooling3D, LocallyConnected1D,
                              LocallyConnected2D, Embedding, Add, Subtract,
                              Multiply, Average, Maximum, Concatenate, Dot,
                              LeakyReLU, PReLU, ELU, ThresholdedReLU,
                              Softmax, BatchNormalization, GaussianNoise,
                              GaussianDropout, AlphaDropout)
    """ Keras and sliding Models """
    from keras.models import Model
    
except:
    from tensorflow.keras.layers import (Dense, Activation, Dropout, Flatten, Input, 
                              Reshape, Permute, RepeatVector, Lambda, 
                              ActivityRegularization, Masking, SpatialDropout1D,
                              SpatialDropout2D, SpatialDropout3D,
                              Conv1D, Conv2D, Conv3D, SeparableConv1D,
                              SeparableConv2D, Conv2DTranspose, 
                              Conv3DTranspose, Cropping1D, Cropping2D,
                              Cropping3D, UpSampling1D, UpSampling2D,
                              UpSampling3D, ZeroPadding1D, ZeroPadding2D,
                              ZeroPadding3D, MaxPooling1D, MaxPooling2D,
                              MaxPooling3D, AveragePooling1D, AveragePooling2D,
                              AveragePooling3D, GlobalMaxPooling1D, 
                              GlobalMaxPooling2D, GlobalMaxPooling3D,
                              GlobalAveragePooling1D, GlobalAveragePooling2D,
                              GlobalAveragePooling3D, LocallyConnected1D,
                              LocallyConnected2D, Embedding, Add, Subtract,
                              Multiply, Average, Maximum, Concatenate, Dot,
                              LeakyReLU, PReLU, ELU, ThresholdedReLU,
                              Softmax, BatchNormalization, GaussianNoise,
                              GaussianDropout, AlphaDropout)

    """ Keras and sliding Models """
    from tensorflow.keras.models import Model


""" Bayesianize """
def _bayesianize(model, weight_regularizer = 1e-6, dropout_regularizer = 1e-5,
                 sliding = False, transfer_weights = True, verbose = False,
                 suffix = '_bayesian', antisuffix = None):
    
    
    """ Loop through layers"""
    input_layers = []
    convertedLayers = dict()
    lookup_table = dict()
    
    nlayers = len(model.layers)
    
    if verbose:
        print('(0%) Bayesianizing model "{}"...'.format(model.name))
    
    for i, layer in enumerate(model.layers):
        """ Get layer class """
        layer_class = layer.__class__.__name__
        layer_name = layer.name
        
        """ Parse to valid class (some classes have different names that their 
            actual wrappers) """
        layer_class = layer_class.replace('InputLayer','Input')
        
        """ Get old layer configuration and change name """
        layer_config = layer.get_config()
        #if ':' in layer_config['name']:
        #    layer_config['name'] = '{}{}:{}'.format(layer_config['name'].split(':')[0],
        #                                              suffix,
        #                                              layer_config['name'].split(':')[1])
        #else:
        layer_config['name'] = '{}{}'.format(layer_config['name'], suffix)
            
            
        """ Now get the true layer object """
        # Input layers are a special case:
        if layer_class == 'Input':
            new_layer = Input(shape = layer_config['batch_input_shape'][1:],
                              name = layer_config['name'],
                              dtype = layer_config['dtype'],
                              sparse = layer_config['sparse'])
            
            new_layer_tensor = new_layer
            msg = '\t({:3.2f}%) - Setting up input layer "{}" renamed '\
                    'to "{}".'.format(100*i/nlayers,
                            layer_name,
                      new_layer.name)
            
            input_layers.append(new_layer)
        
        elif layer_class in ('Add','Concatenate','Subtract', 'Multiply',
                             'Average', 'Maximum', 'Minimum', 'Dot'):
            
            new_layer = eval('{}.from_config(layer_config)'.format(layer_class,
                                                                 layer_config))
            
            # Get list of inputs
            new_layer = new_layer([convertedLayers[ip.name] for ip in layer.input])
            new_layer_tensor = new_layer
        else:
            new_layer_name = layer_config['name']
            layer_config['name'] = layer_config['name'] + '_dropout_inner'
            new_layer = eval('{}.from_config(layer_config)'.format(layer_class,
                                                                 layer_config))
            
            """ Check if we need to add concrete dropout """
            if layer_class in ('Dense', 'Conv1D', 'Conv2D', 'Conv3D',\
                               'SeparableConv1D', 'SeparableConv2D'):
                
                """ Build concretedropout wrapper """
                new_layer = ConcreteDropout(new_layer,
                                            weight_regularizer = weight_regularizer,
                                            dropout_regularizer = dropout_regularizer)
                
                msg = '\t({:3.2f}%) - Added ConcreteDropout to layer "{}" and '\
                          'renamed it to "{}"'.format(100*i/nlayers, 
                                          layer_name,
                                          new_layer_name)
                     
            else:
                msg = '\t({:3.2f}%) - Renamed layer "{}" to "{}".'.format(100*i/nlayers,
                         layer_name,
                          new_layer.name)
                    
            setattr(new_layer,'name',new_layer_name)
            """ Now input tensor to layer"""
            new_layer_tensor = new_layer(convertedLayers[layer.input.name])
                
            
            """ Get old layer weights and biases """
            # check if we can transfer the weights (in case the layer has any)
            plus = ''
            if transfer_weights:
                if hasattr(layer,'get_weights'):
                    w = layer.get_weights()
                    if w != []:
                        try:
                            if not hasattr(new_layer,'layer'):
                                plus = ' Old weights were preserved.'
                                new_layer.set_weights(w)
                            else:
                                plus = ' Old weights were preserved.'
                                new_layer.layer.set_weights(w)
                        except:
                            plus = ' Old weights COULD NOT BE RETRIEVED.'
            msg += plus
            
            
            
            if verbose:
                print(msg)
            
            
        # Save for later linking
        convertedLayers[layer.output.name] = new_layer_tensor
        lookup_table[new_layer.name.split(':')[0].split('/')[0]] = layer_name
        
    """ Finally get the outputs """
    output_layers = [convertedLayers[op.name] for op in model.output] \
                         if isinstance(model.output,list) \
                         else [convertedLayers[model.output.name]]
    
    
    """ And now build the model """
    bayes_model = Model(input_layers, output_layers)
	
    return bayes_model, lookup_table, output_layers
    
""" Debayesianize model """
def _debayesianize(model, sliding = False, verbose = False):
    raise Exception('Function not implemented yet')
    print('deterministic_model')
    
    
