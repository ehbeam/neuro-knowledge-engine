#!/bin/python

import sys
sys.path.append('..')
import prediction

prediction.train_classifier('data-driven', 'reverse', suffix='', clf='_nn')