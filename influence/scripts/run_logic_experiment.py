from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import random

import scipy
import sklearn

import influence.experiments as experiments
from influence.nlprocessor import NLProcessor
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_logic import load_logic

import tensorflow as tf

np.random.seed(42)

data_sets,oneHot2Raw = load_logic()

num_classes = 2

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.0001
# weight_decay = 1000 / len(lr_data_sets.train.labels)
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='spam_logreg')

tf_model.train()

X_train = np.copy(tf_model.data_sets.train.x)
Y_train = np.copy(tf_model.data_sets.train.labels)
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels) 


num_train_examples = Y_train.shape[0] 
num_flip_vals = 6
num_check_vals = 6
num_random_seeds = 40

dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)
fixed_influence_loo_results = np.zeros(dims)
fixed_loss_results = np.zeros(dims)
fixed_random_results = np.zeros(dims)

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))

orig_results = tf_model.sess.run(
    [tf_model.loss_no_reg, tf_model.accuracy_op], 
    feed_dict=tf_model.all_test_feed_dict)
            
print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))


p = np.zeros(len(X_train))
for flips_idx in range(num_flip_vals):
    for random_seed_idx in range(num_random_seeds):
        
        random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)        
        np.random.seed(random_seed)
    
        num_flips = int(num_train_examples / 20) * (flips_idx + 1)    
        idx_to_flip = np.random.choice(num_train_examples, size=num_flips, replace=False)
        Y_train_flipped = np.copy(Y_train)
        Y_train_flipped[idx_to_flip] = 1 #1 - Y_train[idx_to_flip] # calendar 1 publications 0 atis 0
         
        tf_model.update_train_x_y(X_train, Y_train_flipped)
        tf_model.train()        
        flipped_results[flips_idx, random_seed_idx, 1:] = tf_model.sess.run(
            [tf_model.loss_no_reg, tf_model.accuracy_op], 
            feed_dict=tf_model.all_test_feed_dict)
        print('Flipped loss: %.5f. Accuracy: %.3f' % (
                flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))
        
        train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict)
        train_loo_influences = tf_model.get_loo_influences()
        for checks_idx in range(num_check_vals):
            np.random.seed(random_seed + 1)
            num_checks = int(num_train_examples / 20) * (checks_idx + 1)

            print('### Flips: %s, rs: %s, checks: %s' % (num_flips, random_seed_idx, num_checks))

            fixed_loss_results[flips_idx, checks_idx, random_seed_idx, :],idx_to_check \
              = experiments.test_mislabeled_detection_batch_for_logic(
                tf_model, 
                X_train, Y_train,
                Y_train_flipped,
                X_test, Y_test, 
                train_losses, train_loo_influences,
                num_flips, num_checks)
            p[idx_to_check] +=1
            #input("here")
            #for i in idx_to_check:
            #    f.write(oneHot2Raw[i])
                #print('$$$$$$',oneHot2Raw[i])
            #input("here")

theta = 2.0                         # determinism parameter
g = 1-p
ps = p#np.exp(p * theta)
gs = g#np.exp(g * theta)

ps /= np.sum(ps)
gs /= np.sum(gs)

ecdf = np.cumsum(ps)
ecdg = np.cumsum(gs)

f = open('augment-p.txt','w')
f_ = open('augment-1-p.txt','w')
for i in range(100):
    r1=random.uniform(0, 1)
    r2=random.uniform(0, 1)
    x = np.argmin(np.abs(ecdf-r1))    
    y = np.argmin(np.abs(ecdg-r2))    

    f.write(oneHot2Raw[x])
    f_.write(oneHot2Raw[y])
f_.close()
f.close()

np.savez(
    'output/spam_results', 
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results
)
