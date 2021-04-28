#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                              RattlePy Toolbox
 
  Author: Manuel Blanco Valentín 
           (mbvalentin@cbpf.br / manuel.blanco.valentin@gmail.com)

  Collaborators:   Clécio de Bom
                   Luciana Dias

  Sponsors and legal owners: PETROBRÁS
                             CBPF (Centro Brasileiro de Pesquisas Físicas)


  Copyright 2019  
  
  This program is property software; you CANNOT redistribute it and/or modify
  it without the explicit permission of the authors, collaborators and its 
  legal owners (PETROBRÁS and CBPF). Disobeying these guidelines will lead 
  to a violation of the private intellectual property of the owners, along 
  with the legal reprecaussion that this violation might cause.
  
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
ABOUT THE "IO.py" CODE:
    
    The functions defined in this code have the purpose of helping in the
    process of input/output data stream into RattlePy objects.
    
"""


"""
Basic Modules
"""
import os
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

"""
This function checks whether the filepath specified in the model is valid, 
if it exists, if it is necessary to create it, or if it's None.
"""
def _check_filepath(path = None):
    
    # Check if filepath exists
    if not path:
        # filepath does not exist, so let's create it using the model name
        path = os.path.join(os.getcwd())
    # Check if filepath is a valid directory
    if not os.path.isdir(path):
        # Let's try to create it
        try:
            os.mkdir(path)
        except:
            # Let's then try to create any super folders that might be missing
            try:
                super_folders = os.path.normpath(path).split(os.sep)
                for i in np.arange(2,len(super_folders)+1):
                    tmp_file = ''.join([os.sep+p for p in super_folders[1:i]])
                    if not os.path.isdir(tmp_file):
                        os.mkdir(tmp_file)
                        print('Creating folder: %s'%(tmp_file))
            except:
                path = None
                raise ValueError('Error while attempting to create folder: %s to store the model and its results. This might be because python lacks the necessary permissions to create or modify a folder on the specified path. Please, indicate a different folder where python might have writting permissions, or create the specified path manually in your system window explorer.')
            
""" function to save results """
def save_results(results, path = os.getcwd()):
    with open(os.path.join(path,'results.pkl'),'wb') as f:
        pickle.dump(results,f)

""" function to load results """
def load_results(path = os.getcwd()):
    with open(os.path.join(path,'results.pkl'),'rb') as f:
        results = pickle.load(f)
    
    return results
        