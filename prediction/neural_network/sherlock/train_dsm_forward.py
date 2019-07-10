#!/bin/python

import sys
sys.path.append('..')
import prediction

prediction.train_classifier('dsm', 'forward', suffix='_opsim', clf='', use_hyperparams=True)