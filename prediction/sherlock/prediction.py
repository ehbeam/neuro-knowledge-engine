#!/usr/bin/python3

import os, pickle, random
import pandas as pd
import numpy as np
from collections import OrderedDict
np.random.seed(42)

from sklearn.preprocessing import binarize
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from time import time


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_doc_term_matrix(version=190325, binarize=True):
	dtm = pd.read_csv("../../data/text/dtm_{}.csv.gz".format(version), compression="gzip", index_col=0)
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm


def load_framework(framework, suffix=""):
	list_file = "../../ontology/lists/lists_{}{}.csv".format(framework, suffix)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "../../ontology/circuits/circuits_{}.csv".format(framework)
	circuits = pd.read_csv(circuit_file, index_col=0)
	return lists, circuits


def score_lists(lists, dtm, label_var="LABEL"):
	dtm = pd.DataFrame(binarize(dtm, threshold=0), index=dtm.index, columns=dtm.columns)
	labels = OrderedDict.fromkeys(lists[label_var])
	list_counts = pd.DataFrame(index=dtm.index, columns=labels)
	for label in list_counts.columns:
		tkns = lists.loc[lists[label_var] == label, "TOKEN"]
		tkns = [token for token in tkns if token in dtm.columns]
		list_counts[label] = dtm[tkns].sum(axis=1)
	list_scores = doc_mean_thres(list_counts)
	return list_scores


def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results["rank_test_score"] == i)
		for candidate in candidates:
			print("-" * 30 + "\nModel rank: {0}".format(i))
			print("Cross-validated score: {0:.4f} (std: {1:.4f})".format(
				  results["mean_test_score"][candidate],
				  results["std_test_score"][candidate]))
			print("Parameters: {0}".format(results["params"][candidate]))


def search_grid(X, Y, param_grid, scoring="roc_auc", n_iter=25):
	clf = OneVsRestClassifier(MLPClassifier())
	grid_search = RandomizedSearchCV(clf, param_grid, n_iter=n_iter, scoring=scoring, cv=5, n_jobs=5)
	start = time()
	grid_search.fit(X, Y)
	print("\nGridSearchCV took %.2f seconds for %d candidate parameter settings\n"
		  % (time() - start, len(grid_search.cv_results_["params"])))
	report(grid_search.cv_results_)
	return grid_search


def train_classifier(framework, direction, suffix=""):

	fit_file = "../fits/{}_{}.p".format(framework, direction)
	if not os.path.isfile(fit_file):

		act_bin = pd.read_csv("../../data/brain/coordinates.csv", index_col=0)
		dtm_bin = load_doc_term_matrix(version=version, binarize=True)

		train = [int(pmid.strip()) for pmid in open("../../data/splits/train.txt")]

		lists, circuits = load_framework(framework, suffix=suffix)
		scores = score_lists(lists, dtm, label_var=level.upper())
			
		param_grid = {"estimator__hidden_layer_sizes": [tuple([50]*n_layers) for n_layers in range(1,6)],
					  "estimator__activation": ["logistic", "tanh", "relu"],
					  "estimator__solver": ["lbfgs", "sgd", "adam"],
					  "estimator__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
					  "estimator__random_state": [42], 
					  "estimator__max_iter": [100]}

		if direction == "forward":
			cv_results = search_grid(scores.loc[train], act_bin.loc[train], param_grid, 
									 scoring="roc_auc", n_iter=100)
		elif directon == "reverse":
			cv_results = search_grid(act_bin.loc[train], scores.loc[train], param_grid, 
									 scoring="roc_auc", n_iter=100)

		top_clf = cv_results.best_estimator_
		pickle.dump(top_clf, open(fit_file, "wb"), protocol=2)

