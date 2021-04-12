#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:53:20 2019

@author: mbvalentin
"""

""" Basic modules """

import os, platform
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
import numpy as np
import getpass, datetime
import matplotlib
matplotlib.use('Agg')

""" Data augmentation """
from scipy.ndimage.filters import gaussian_filter1d as gf1d

""" Json handling """
import json

""" Load a catalog from json file """
def load_catalog(filename):
    Data = json.load(open(filename))
    
    # transform data into np array (json parser loads them as lists)
    for c in Data['catalogs']:
        for ch in Data['catalogs'][c]['channels']:
            try:
                float(Data['catalogs'][c]['channels'][ch]['data'][0][0])
                xx = np.array(Data['catalogs'][c]['channels'][ch]['data'])
                nanid = np.where(xx == 'nan')
                xx[nanid] = 0
                Data['catalogs'][c]['channels'][ch]['data'] = xx.astype(np.float)
                Data['catalogs'][c]['channels'][ch]['data'][nanid] = np.nan
            except:
                print('Leaving channel {} as imported'.format(ch))
            
    return Data


""" This function is used to transform a dictionary object into a json formatted text """
def _recursivity(dct, level, convert = True):
        
    # number of tabs:
    tab = ''.join(['\t']*level)
        
    txt = ''
    
    # loop through all entries in dct
    if isinstance(dct,dict):
        
        txt = tab
        # we shall put this entry between brackets
        tab_b = ''.join(['\t']*(level+1))
        # tabulation one level further
        tab_e = ''.join(['\t']*(level+2))
        
        txt += '\n' + tab_e + '{\n'
        for e in dct:
            print('{}{} - {}'.format(tab_e,level+2,e))
            
            # add entry
            #txt += tab_b + '{\n'
            txt += '{}"{}":'.format(tab_e, e)
            
            # recursivity on dictionary
            txt += _recursivity(dct[e], level + 1)
            
            # remove last '\n' and close entry
            txt = txt[:-1] + ',\n'
            #txt += tab_e + '},'
        
        # discard last comma and place line jump
        txt = txt[:-2]
        txt += '\n'
        
        # close bracket
        txt += tab_e + '}\n'
            
    elif isinstance(dct,str) or isinstance(dct,int) or isinstance(dct,float):
        tab_b = ''.join(['\t']*(level+2))
        # simply a comment or a string value of some sort
        txt += '"{}"\n'.format(dct)
        print('{}{} - "{}"'.format(tab_b,level+2,dct))
    elif isinstance(dct,np.ndarray) or isinstance(dct,list) or isinstance(dct,tuple):
        tab_b = ''.join(['\t']*(level+2))
        dct = np.array(dct)
        print('{}{} - array with shape {}'.format(tab_b,level+2,dct.shape))
        if convert:
            data_tmp = np.array2string(dct,separator=',',threshold=np.inf).replace('. ','.0').replace('\n','')
            txt += '{}\n'.format(data_tmp).replace('.]','.0]')
        
    
    return txt

""" Display information contained inside a catalog dictionary """
def print_catalog(obj):
    _recursivity(obj, -2, convert = False)


""" Convert dictionary object to json file """
def dict_to_catalog(obj, filename, extension = '.json'):
    txt = _recursivity(obj, -2)
    txt = txt.replace("'",'"')
    with open('{}{}'.format(filename.replace('.json',''),extension),'w') as f:
        f.write(txt)

def build_catalog(outputs, predictions, 
                  logvars = None,
                  correct_aleatoric_error = True,
                  apply_linear_correction = True,
                  confidence_intervals = [.68, .95, .99],
                  sample_ids = None,
                  metadata = None,
                  filename = None, 
                  catalog_name = 'Unknown Catalog',
                  save_to_path = os.getcwd() ):
    
    """ Parse confidence intervals """
    confidence_intervals = np.array(confidence_intervals)
    if np.any(confidence_intervals > 1):
        confidence_intervals /= 100
        print('WARNING! confidence intervals should be specified between 0.5 '\
              'and 1.0, but found numbers greater than 1. Dividing by 100 and continyuing.')
    if np.any(confidence_intervals < .5):
        confidence_intervals = [ci for ci in confidence_intervals if ci > 0.5]
        print('WARNING! Confidence intervals should be specified between 0.5 '\
              'and 1.0, outside these boundaries. Keeping only valid values and continuying.')
    if len(confidence_intervals) == 0:
        confidence_intervals = [.68, .95, .99]
    
    
    catalog = dict()
    
    """ Basic info """
    catalog['name'] = filename
    catalog['creation_date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    catalog['creator'] = getpass.getuser()
    catalog['system_info'] = 'OS {} - Platform {} - '\
            'Version {} - '\
            'Processor {}'.format(platform.platform(),
                                    platform.machine(),
                                    platform.version(),
                                    platform.processor()
                                    )
    
    """ Now build the catalogs """
    catalog['catalogs'] = dict()
    for c in outputs:
        channels = dict()
        if sample_ids is not None:
            channels['sample_id'] = {'info':'',
                                       'unit':'',
                                       'description':'',
                                       'data':sample_ids}
            
        """ First and most important thing, let's calculate the bayesian 
            uncertainties, because if we have the aleatoric error and if
            flag "correct uncertainties" is set to True we will have to 
            correct the posteriors (and all the following calculations depend)
            on posteriors...
        """
            
        preds = predictions[c]
        channels['bayesian_uncertainties'] = dict()
        
        if isinstance(logvars,dict):
            if c in logvars:
                if logvars[c] is not None:
                    channels['logvars'] = {'info':'',
                                   'unit':'',
                                   'description':'LogVars of {} values using {} iterations'.format(c,len(predictions[c])),
                                   'data':logvars[c]}
                    
                    # estimate aleatoric error
                    aleatoric = np.exp(logvars[c].mean(0))
                    
                    # set aleatoric error in place
                    channels['bayesian_uncertainties']['aleatoric'] = aleatoric
                    
                    """ Correct posteriors/medians if required """
                    if correct_aleatoric_error:
                        preds = np.random.normal(loc=preds, 
                                                 scale=np.exp(logvars[c]))
        
        """ In case we need to apply linear correction, let's """
        preds_info = ''
        if apply_linear_correction:
            """ Try to correct bias baseline"""
            idxs = None
            if 'indexes' in metadata:
                if 'training' in metadata['indexes']:
                    idxs = metadata['indexes']['training']
                    
            preds, lin_adj_coeffs = linear_adjust({c:preds}, {c:outputs[c]}, \
                                                  indexes = idxs, \
                                                  kind = 'mean')
            preds = preds[c]
            lin_adj_coeffs = lin_adj_coeffs[c]
            preds_info = 'Corrected using linear_adjust_coefficients: {}'.format(lin_adj_coeffs)
        
        # Predictive and epistemic errors
        predictive = (((preds - outputs[c][None,:])**2).mean(0))**0.5
        epistemic = preds.var(0)
        channels['bayesian_uncertainties']['predictive'] = predictive
        channels['bayesian_uncertainties']['epistemic'] = epistemic
        
        """ Actuals """
        channels['actuals'] = {'info':'',
                               'unit':'',
                               'description':'Original (ground-truth) {} values'.format(c),
                               'data':outputs[c]}
        
        """ Medians """
        channels['medians'] = {'info':'',
                               'unit':'',
                               'description':'Medians of the predicted {} values'.format(c),
                               'data':preds.mean(0)}
        """ Confidence intervals """
        channels['intervals'] = {'info':'',
                               'unit':'',
                               'description':'Confidence intervals {} of the predicted {} values'.format(','.join(['{:3.0f}%'.format((100*ci)) for ci in confidence_intervals]),
                                                                   c),
                               'data':dict()}
        for ci in confidence_intervals:
            lb, ub = np.percentile(preds, 
                                   [(1-ci)/2*100, (1+ci)/2*100], 
                                   axis = 0)
            channels['intervals']['data']['{:d}%'.format(int(100*ci))] = {'lower': lb,
                                                                  'upper': ub}
        
        """ posteriors """
        channels['posteriors'] = {'info':'',
                               'unit':'',
                               'description':'Estimations of {} values using {} iterations'.format(c,len(predictions[c])),
                               'data':preds}
        

        """ Set this whole structure in place """
        catalog['catalogs'][c] = {'info':preds_info,
                                   'channels':channels}
    
    """ Add metadata """
    if metadata is not None:
        catalog['metadata'] = metadata
    
    """ Store file in disk, in case required """
    # if filename is not None and save_to_path is not None:
        # dict_to_catalog(catalog, os.path.join(save_to_path,filename))
    
    return catalog


""" Normalize data """
def normalize_data(x, norm_coefficients = None):
    if norm_coefficients is None:
        norm_coefficients = {xn: (x[xn].min(), x[xn].max()) for xn in x}
    x_norm = {xn: (x[xn]-norm_coefficients[xn][0])/(norm_coefficients[xn][1]-norm_coefficients[xn][0]) for xn in norm_coefficients}
    return x_norm, norm_coefficients

""" Denormalize data back to original range and bias """
def denormalize_data(x_norm, norm_coefficients, prefun = None, postfun = None, 
                     apply_bias = True):
    
    if prefun is None:
        prefun = lambda x: x
    if postfun is None:
        postfun = lambda x: x
    
    x_denorm = {xn: prefun(x_norm[xn])*(norm_coefficients[xn][1]-norm_coefficients[xn][0]) \
                for xn in x_norm}
    if apply_bias:
        x_denorm = {xn: x_denorm[xn] + norm_coefficients[xn][0] \
                for xn in x_denorm}
    x_denorm = {xn: postfun(x_denorm[xn]) for xn in x_denorm}
    
    return x_denorm

""" Apply linear adjust to try to correct bias """
def linear_adjust(posts, outs, lin_adj_coeffs = None, indexes = None, kind = 'mean'):        
    if lin_adj_coeffs is None:
        if kind == 'mean':
            if indexes is not None:
                indexes = np.array(indexes)
                xx = {xn: posts[xn].mean(0)[indexes,0] for xn in posts}
                yy = {xn: outs[xn][indexes,0] for xn in outs}
            else:
                xx = {xn: posts[xn].mean(0)[:,0] for xn in posts}
                yy = {xn: outs[xn][:,0] for xn in outs}
        
        elif kind == 'all':
            if indexes is not None:
                indexes = np.array(indexes)
                xx = {xn: posts[xn][indexes,0].ravel() for xn in posts}
                yy = {xn: ([outs[xn][indexes,0]]*len(posts[xn])).ravel() for xn in outs}
            else:
                xx = {xn: posts[xn][:,0].ravel() for xn in posts}
                yy = {xn: ([outs[xn][:,0]]*len(posts[xn])).ravel() for xn in outs}
        else:
            raise ValueError('Undefined specified kind: "{}". Valid kinds are '\
                             '"mean" or "all".'.format(kind))
        
        # Fit line
        lin_adj_coeffs = {xn: np.polyfit(xx[xn], yy[xn], 1) for xn in outs}
        
    # Apply correction
    post_norm_adj = {xn: np.poly1d(lin_adj_coeffs[xn])(posts[xn]) for xn in outs}
    
    return post_norm_adj, lin_adj_coeffs


""" Data augmentation by smoothing the specified inputs """
def data_augmentation(inputs, outputs, 
                      input_channels = [], 
                      kernels = (3,5,7,11),
                      noises = None):
    
    if not isinstance(input_channels ,list):
        input_channels = [input_channels]
        
    input_channels = [ip for ip in input_channels if ip in list(inputs)]
    # apply
    repeat_channels = [ip for ip in list(inputs) if ip not in input_channels]
    
    inputs_ = {ic: np.vstack((inputs[ic],) + \
                             tuple([gf1d(inputs[ic], k, axis = 1) \
                                    for k in kernels])) \
                for ic in input_channels}
    
    inputs_ = dict(inputs_, **{ic: np.vstack((inputs[ic],) + \
                                             tuple([inputs[ic] \
                                                    for k in kernels])) \
                                for ic in repeat_channels})
    
    if noises is not None:
        inputs__ = {ic: np.vstack(tuple([inputs[ic] + \
                                         np.random.normal(loc=0.0, scale=n, size = inputs[ic].shape) \
                                    for n in noises])) \
                for ic in input_channels}
        
        inputs__ = dict(inputs__, **{ic: np.vstack(tuple([inputs[ic] \
                                                    for n in noises])) \
                                for ic in repeat_channels})
    
        inputs_ = {ic: np.vstack((inputs_[ic], inputs__[ic])) for ic in inputs_}
    
    # Repeat outputs
    if outputs is not None:
        outputs_ = {oc: np.vstack((outputs[oc],) + \
                                 tuple([outputs[oc] \
                                        for k in kernels])) \
                    for oc in list(outputs)}
        
        if noises is not None:
            outputs_ = {oc: np.vstack((outputs_[oc],) + \
                                 tuple([outputs[oc] \
                                        for n in noises])) \
                    for oc in list(outputs)}
    
    return inputs_, outputs_