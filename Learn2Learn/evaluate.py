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
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.learn.python.learn import monitored_session as ms

import meta
import util


tf.reset_default_graph()

flags = tf.flags
logging = tf.logging


path = './testseagle'
num_epochs = 2
log_period = 10
evaluation_period = 1000
evaluation_epochs = 20
problem = 'SEAGLE'

learning_rate = 0.1
second_derivatives = False 
if(False):
    optimizer = 'Adam'#'L2L'
    learning_rate = 0.1
    num_steps = 20

else:
    num_steps = 100
    optimizer = 'L2L'
    seed = None
    num_epochs = 1
    num_unrolls = 50


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
    update = optimizer.minimize(cost_op) # argmin(*)
    reset = [problem_reset, optimizer_reset]

elif optimizer == "L2L":
    if path is None:
        logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(**net_config)
    meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
    _, update, reset, cost_op, problem_vars = meta_loss

else:
    raise ValueError("{} is not a valid optimizer".format(optimizer))

sess = ms.MonitoredSession()# as sess:
# Prevent accidental changes to the graph.
tf.get_default_graph().finalize()

total_time = 0
total_cost = 0

for i in xrange(num_epochs):
    # Training.
    time, cost, problem_res = util.run_epoch_test(sess, cost_op, problem_vars, [update], reset, num_unrolls)
    total_time += time
    total_cost += cost
    print('Cost at Iteration: '+str(i)+' is: '+str(total_cost))

# Results.
util.print_stats("Epoch {}".format(num_epochs), total_cost,
             total_time, num_epochs)


#%% print the results 
# not quiet clear how the shaping is done..
# e.g. problem_res.shape = (20, 2, 100, 4, 100) # (nsteps, (real/imag), nx, nz, ny) - > REAL/IMAG are two seperate variables which we try to optimize here 
for i in range(problem_res.shape[0]):
    mycurrent_result = np.abs(problem_res[i,0,:,1,:]+1j*problem_res[i,1,:,1,:])
    plt.imshow(np.abs(np.squeeze(mycurrent_result))), plt.show()