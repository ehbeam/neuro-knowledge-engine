#!/bin/python

import sys
sys.path.append('..')
import prediction

prediction.train_classifier('data-driven', 'forward', suffix='', clf='_nn', use_hyperparams=True)