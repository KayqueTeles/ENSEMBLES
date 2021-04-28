#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 22:07:42 2019

@author: mbvalentin
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

""" Basic modules """
import threading

""" class based on thread , which can be stopped"""
class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self, **kwargs):
        super(StoppableThread, self).__init__(**kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()