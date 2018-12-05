#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:05:04 2018

@author: bene
"""

import tensorflow as tf
import numpy as np 

my_size = 2
my_f = tf.constant([1, 2], tf.float32)
my_greens = tf.constant([[1, 2], [3, 4]], tf.float32)
my_u = tf.constant([1, 2], tf.float32)
my_u = tf.expand_dims(my_u, axis = 1)

my_uin = tf.constant([1, 2], tf.float32)
my_uin = tf.expand_dims(my_uin, axis = 1)

my_I = tf.eye(my_size, my_size)
my_diagf = tf.diag(my_f, name=None)

my_ud = tf.matmul((my_I-tf.matmul(my_greens, my_diagf)), my_u) - my_uin

