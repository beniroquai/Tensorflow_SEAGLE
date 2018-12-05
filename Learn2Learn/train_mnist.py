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


save_path = './testQuadratic'
num_epochs=10000
log_period=100
evaluation_period=1000
evaluation_epochs=20
problem='quadratic'
optimizer='L2L' 
num_steps=100
unroll_length=20
learning_rate=0.001
second_derivatives=False


# Configuration.
num_unrolls = num_steps // unroll_length

if True:
    if os.path.exists(save_path):
        raise ValueError("Folder {} already exists".format(save_path))
    else:
      os.mkdir(save_path)
    
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

sess = ms.MonitoredSession()

      # Prevent accidental changes to the graph.
tf.get_default_graph().finalize()

best_evaluation = float("inf")
total_time = 0
total_cost = 0
for e in xrange(num_epochs):
  # Training.
  time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                      num_unrolls)
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
  for f in os.listdir(save_path):
    os.remove(os.path.join(save_path, f))
  print("Saving meta-optimizer to {}".format(save_path))
  optimizer.save(sess, save_path)
  best_evaluation = eval_cost