# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util

flags = tf.flags
logging = tf.logging

tf.reset_default_graph()


save_path = './testseagle' #'./testdeconv' 
num_epochs = 100000
log_period = 10
evaluation_period = 1000
evaluation_epochs = 1
problem = 'SEAGLE'# 'SEAGLE'#'deconv'#
num_steps = 100 # number of steps the optimizer should train the optmizee (a new random function will be updated for num_steps in each epoch)
unroll_length = 100 # number of steps after which the optimizer gets updated 
learning_rate = 0.001
second_derivatives = False 
eval_cost = 0

## START PROGRAMM HERE

# Configuration.
num_unrolls = num_steps // unroll_length

if save_path is not None:
    if ~os.path.exists(save_path):
        try:
            os.mkdir(save_path)
        except(FileExistsError):
            print('Dir exist!')

# Problem.
problem, net_config, net_assignments = util.get_config(problem)

# Optimizer setup.
optimizer = meta.MetaOptimizer(**net_config)
minimize = optimizer.meta_minimize(
    problem, unroll_length,
    learning_rate=learning_rate,
    net_assignments=net_assignments,
    second_derivatives=second_derivatives)
step, update, reset, cost_op, _ = minimize

with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()
    
    best_evaluation = float("inf")
    total_time = 0
    total_cost = 0
    for e in xrange(num_epochs):
        # Training.
        time, cost = util.run_epoch(sess, cost_op, [update, step], reset, num_unrolls)
        total_time += time
        total_cost += cost
        
        # Logging.
        if (e + 1) % log_period == 0:
            util.print_stats("Epoch {}".format(e + 1), total_cost, total_time,
                         log_period)
        total_time = 0
        total_cost = 0
        
        # Evaluation.
        if (e + 1) % evaluation_period == 0:
            eval_cost = 0
            eval_time = 0
            for _ in xrange(evaluation_epochs):
                time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                          num_unrolls)
                eval_time += time
                eval_cost += cost
            
            util.print_stats("EVALUATION", eval_cost, eval_time,
                             evaluation_epochs)
        
        if save_path is not None and eval_cost < best_evaluation:
            print("Removing previously saved meta-optimizer")
            #os.remove(os.path.join(save_path, f))
            print("Saving meta-optimizer to {}".format(save_path))
            optimizer.save(sess, save_path)
            best_evaluation = eval_cost
