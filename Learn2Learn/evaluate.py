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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util

flags = tf.flags
logging = tf.logging


path = './test'
num_epochs = 100
log_period = 10
evaluation_period = 1000
evaluation_epochs = 20
problem = 'simple'
num_steps = 100
unroll_length = 100
learning_rate = 0.001
second_derivatives = False 
optimizer = 'L2L'
seed = None



# Configuration.
num_unrolls = num_steps

if seed:
    tf.set_random_seed(seed)

# Problem.
problem, net_config, net_assignments = util.get_config(problem, path)

# Optimizer setup.
if optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]

elif optimizer == "L2L":
    if path is None:
        logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(**net_config)
    meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
    _, update, reset, cost_op, _ = meta_loss

else:
    raise ValueError("{} is not a valid optimizer".format(optimizer))

with ms.MonitoredSession() as sess:
    # Prevent accidental changes to the graph.
    tf.get_default_graph().finalize()

    total_time = 0
    total_cost = 0
    for _ in xrange(num_epochs):
        # Training.
        time, cost = util.run_epoch(sess, cost_op, [update], reset,
                          num_unrolls)
        total_time += time
        total_cost += cost

# Results.
util.print_stats("Epoch {}".format(num_epochs), total_cost,
             total_time, num_epochs)
