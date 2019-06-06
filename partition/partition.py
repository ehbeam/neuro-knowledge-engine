#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)
from scipy.spatial.distance import cdist
from matplotlib import font_manager, rcParams
from collections import OrderedDict


arial = "../style/Arial Unicode.ttf"
rcParams["axes.linewidth"] = 1


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_coordinates():
	atlas_labels = pd.read_csv("../data/brain/labels.csv")
	activations = pd.read_csv("../data/brain/coordinates.csv", index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]
	return activations


def load_doc_term_matrix(version=190124, binarize=True):
	dtm = pd.read_csv("../data/text/dtm_{}.csv.gz".format(version), compression="gzip", index_col=0)
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm


def load_lexicon(sources):
	lexicon = []
	for source in sources:
		file = "../lexicon/lexicon_{}.txt".format(source)
		lexicon += [token.strip() for token in open(file, "r").readlines()]
	return sorted(lexicon)


def load_framework(framework, suffix=""):
	list_file = "../ontology/lists/lists_{}{}.csv".format(framework, suffix)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "../ontology/circuits/circuits_{}.csv".format(framework)
	circuits = pd.read_csv(circuit_file, index_col=0)
	return lists, circuits


def plot_partition(framework, doc_dists, transitions):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams

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

