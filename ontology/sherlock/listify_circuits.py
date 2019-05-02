#!/usr/bin/python3

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import RandomizedSearchCV
from time import time

def doc_mean_thres(df):
  doc_mean = df.mean()
  df_bin = 1.0 * (df.values > doc_mean.values)
  df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
  return df_bin

def load_doc_term_matrix(version=190124, binarize=True):
  dtm = pd.read_csv("../../data/text/dtm_{}.csv.gz".format(version), compression="gzip", index_col=0)
  if binarize:
    dtm = doc_mean_thres(dtm)
  return dtm

def load_lexicon(sources):
  lexicon = []
  for source in sources:
    file = "../../lexicon/lexicon_{}.txt".format(source)
    lexicon += [token.strip() for token in open(file, "r").readlines()]
  return sorted(lexicon)

def load_domains(k):
  list_file = "../lists/lists_k{:02d}_oplen.csv".format(k)
  lists = pd.read_csv(list_file, index_col=None)
  circuit_file = "../circuits/circuits_k{:02d}.csv".format(k)
  circuits = pd.read_csv(circuit_file, index_col=None)
  return lists, circuits

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

np.random.seed(42)

act_bin = pd.read_csv("../../data/brain/coordinates.csv", index_col=0)

version = 190325
dtm = load_doc_term_matrix(version=version, binarize=False)
dtm_bin = doc_mean_thres(dtm)

lexicon = load_lexicon(["cogneuro"])
lexicon = sorted(list(set(lexicon).intersection(dtm_bin.columns)))
dtm_bin = dtm_bin[lexicon]

train = [int(pmid.strip()) for pmid in open("../../data/splits/train.txt")]

param_grid = {"estimator__hidden_layer_sizes": [tuple([50]*n_layers) for n_layers in range(1,6)],
                  "estimator__activation": ["logistic", "tanh", "relu"],
                  "estimator__solver": ["lbfgs", "sgd", "adam"],
                  "estimator__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                  "estimator__random_state": [42], 
                  "estimator__max_iter": [100]}
 
def optimize_circuits(k):
  print("Assessing k={:02d} circuits".format(k))

  lists, circuits = load_domains(k)

  function_features = pd.DataFrame(index=dtm_bin.index, columns=range(1, k+1))
  structure_features = pd.DataFrame(index=act_bin.index, columns=range(1, k+1))
  for i in range(1, k+1):
      functions = lists.loc[lists["CLUSTER"] == i, "TOKEN"]
      function_features[i] = dtm_bin[functions].sum(axis=1)
      structures = circuits.loc[circuits["CLUSTER"] == i, "STRUCTURE"]
      structure_features[i] = act_bin[structures].sum(axis=1)
  function_features = pd.DataFrame(doc_mean_thres(function_features)
                                   index=dtm_bin.index, columns=range(1, k+1))
  structure_features = pd.DataFrame(binarize(structure_features), 
                                   index=act_bin.index, columns=range(1, k+1))

  forward_cv = search_grid(function_features.loc[train], structure_features.loc[train], param_grid)
  forward_clf = forward_cv.best_estimator_
  forward_file = "fits/forward_k{:02d}_oplen.p".format(k)
  pickle.dump(forward_clf, open(forward_file, "wb"), protocol=2)

  reverse_cv = search_grid(structure_features.loc[train], function_features.loc[train], param_grid)
  reverse_clf = reverse_cv.best_estimator_
  reverse_file = "fits/reverse_k{:02d}_oplen.p".format(k)
  pickle.dump(reverse_clf, open(reverse_file, "wb"), protocol=2)
  