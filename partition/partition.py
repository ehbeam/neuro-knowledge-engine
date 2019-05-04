#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
from utilities import *

from matplotlib import font_manager, rcParams
rcParams["axes.linewidth"] = 1


def plot_partition(framework, doc_dists, transitions):

	import matplotlib.pyplot as plt
	from matplotlib import cm

	fig = plt.figure(figsize=(10,10), frameon=False)
	ax = fig.add_axes([0,0,1,1])

	X = doc_dists.values.astype(np.float)
	im = ax.matshow(X, cmap=cm.Greys_r, vmin=0, vmax=1, alpha=1) 
	plt.xticks(transitions)
	plt.yticks(transitions)
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	plt.savefig("figures/partition_{}.png".format(framework), 
				dpi=250, bbox_inches="tight")
	plt.show()

