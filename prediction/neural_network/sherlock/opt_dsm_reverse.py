#!/bin/python

import sys
sys.path.append('..')
import prediction

prediction.train_classifier('dsm', 'reverse', suffix='_opsim', clf='')