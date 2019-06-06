#!/usr/bin/python

import os, pickle, random
import pandas as pd
import numpy as np
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_coordinates():
	atlas_labels = pd.read_csv("../../data/brain/labels.csv")
	activations = pd.read_csv("../../data/brain/coordinates.csv", index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]
	return activations


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


def score_lists(lists, dtm_bin, label_var="DOMAIN"):
	labels = OrderedDict.fromkeys(lists[label_var])
	list_counts = pd.DataFrame(index=dtm_bin.index, columns=labels)
	for label in list_counts.columns:
		tkns = lists.loc[lists[label_var] == label, "TOKEN"]
		list_counts[label] = dtm_bin[tkns].sum(axis=1)
	list_scores = doc_mean_thres(list_counts)
	return list_scores


def optimize_hyperparameters(param_list, train_set, val_set, max_iter=500):
	
	op_score_val, op_fit = 0, 0
	
	for params in param_list:

		print("-" * 75)
		print("   ".join(["{} {}".format(k.upper(), v) for k, v in params.items()]))
		print("-" * 75 + "\n")

		# Specify the classifier with the current hyperparameter combination
		classifier = OneVsRestClassifier(LogisticRegression(penalty=params["penalty"], C=params["C"], fit_intercept=params["fit_intercept"], 
														max_iter=max_iter, tol=1e-10, solver="liblinear", random_state=42))

		# Fit the classifier on the training set
		classifier.fit(train_set[0], train_set[1])

		# Evaluate on the validation set
		preds_val = classifier.predict_proba(val_set[0])
		score_val = roc_auc_score(val_set[1], preds_val, average="macro")
		print("\n   Validation Set ROC-AUC {:6.4f}\n".format(score_val))
		
		# Update outputs if this model is the best so far
		if score_val > op_score_val:
			print("   Best so far!\n")
			op_score_val = score_val
			op_fit = classifier

	return op_fit


def train_classifier(framework, direction, suffix="", dtm_version=190325):

	fit_file = "fits/{}_{}.p".format(framework, direction)
	if not os.path.isfile(fit_file):

		# Load the data splits
		splits = {}
		for split in ["train", "validation"]:
			splits[split] = [int(pmid.strip()) for pmid in open("../../data/splits/{}.txt".format(split), "r").readlines()]

		# Load the activation coordinate and text data
		act_bin = load_coordinates()
		dtm_bin = load_doc_term_matrix(version=dtm_version, binarize=True)
		
		# Score the texts using the framework
		lists, circuits = load_framework(framework, suffix=suffix)
		scores = score_lists(lists, dtm_bin)
			
		# Specify the hyperparameters for the randomized grid search
		param_grid = {"penalty": ["l1", "l2"],
				  	  "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
					  "fit_intercept": [True, False]}
		param_list = list(ParameterSampler(param_grid, n_iter=28, random_state=42))
		max_iter = 1000

		# Split the train set into batches and load the validation set as a batch
		if direction == "forward":
			train_set = [scores.loc[splits["train"]], act_bin.loc[splits["train"]]]
			val_set = [scores.loc[splits["validation"]], act_bin.loc[splits["validation"]]]

		elif direction == "reverse":
			train_set = [act_bin.loc[splits["train"]], scores.loc[splits["train"]]]
			val_set = [act_bin.loc[splits["validation"]], scores.loc[splits["validation"]]]

		# Search for the optimal hyperparameter combination
		op_fit = optimize_hyperparameters(param_list, train_set, val_set, max_iter=max_iter)

		# Export the optimized results
		pickle.dump(op_fit, open(fit_file, "wb"), protocol=2)
		

train_classifier("data-driven", "forward", suffix="", dtm_version=190325)
train_classifier("data-driven", "reverse", suffix="", dtm_version=190325)
train_classifier("rdoc", "forward", suffix="_opsim", dtm_version=190325)
train_classifier("rdoc", "reverse", suffix="_opsim", dtm_version=190325)
train_classifier("dsm", "forward", suffix="_opsim", dtm_version=190325)
train_classifier("dsm", "reverse", suffix="_opsim", dtm_version=190325)
