#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)


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


def load_framework(framework, suffix=""):
	list_file = "../ontology/lists/lists_{}{}.csv".format(framework, suffix)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "../ontology/circuits/circuits_{}.csv".format(framework)
	circuits = pd.read_csv(circuit_file, index_col=0)
	return lists, circuits