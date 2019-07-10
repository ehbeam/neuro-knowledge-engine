#!/bin/python

import sys
sys.path.append('..')
import prediction

prediction.train_classifier('rdoc', 'reverse', suffix='_opsim', clf='')