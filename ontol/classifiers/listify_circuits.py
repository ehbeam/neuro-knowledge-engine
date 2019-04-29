#!/usr/bin/python3

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import binarize
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from time import time

def mean_thres(df):
  col_mean = df.mean()
  df_bin = np.empty((df.shape[0], df.shape[1]))
  i = 0
  for col, doc_mean in col_mean.iteritems():
    df_bin[:,i] = 1 * (df[col] > doc_mean)
    i += 1
  df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
  return df_bin

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("-" * 30 + "\nModel rank: {0}".format(i))
            print("Cross-validated score: {0:.4f} (std: {1:.4f})".format(
                  results["mean_test_score"][candidate],
                  results["std_test_score"][candidate]))
            print("Parameters: {0}".format(results["params"][candidate]))

def search_grid(X, Y, param_grid, scoring="roc_auc"):
    clf = OneVsRestClassifier(LogisticRegression())
    grid_search = GridSearchCV(clf, param_grid, scoring=scoring, cv=5, n_jobs=5)
    start = time()
    grid_search.fit(X, Y)
    print("\nGridSearchCV took %.2f seconds for %d candidate parameter settings\n"
          % (time() - start, len(grid_search.cv_results_["params"])))
    report(grid_search.cv_results_)
    return grid_search

np.random.seed(42)

act_bin = pd.read_csv("../../data/dcm_0mm_thres0.csv", index_col=0)

rdoc_version = 190124
dtm = pd.read_csv("../../data/dtm_{}.csv.gz".format(rdoc_version), compression="gzip", index_col=0)
lexicon = [tkn.strip() for tkn in open("../../data/lexicon/lexicon_cogneuro_preproc.txt", "r") if tkn.strip() in list(dtm.columns)]
rdoc_seeds = [seed for seed in set(pd.read_csv("../../data/rdoc_{}/rdoc_seeds.csv".format(rdoc_version))["TOKEN"]) if seed in dtm.index]
lexicon = sorted(list(set(lexicon + rdoc_seeds)))
dtm = dtm[lexicon]

dtm_bin = np.empty((dtm.shape[0], dtm.shape[1]))
for i, (word, mean_docfreq) in enumerate(dtm.mean().iteritems()):
    dtm_bin[:,i] = 1 * (dtm[word] > mean_docfreq)
dtm_bin = pd.DataFrame(dtm_bin, index=dtm.index, columns=dtm.columns)

train = [int(pmid.strip()) for pmid in open("../../data/splits/train.txt")]
val = [int(pmid.strip()) for pmid in open("../../data/splits/validation.txt")]

param_grid = {"estimator__penalty": ["l1", "l2"],
                  "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "estimator__fit_intercept": [True, False],
                  "estimator__solver": ["liblinear"],
                  "estimator__random_state": [42], 
                  "estimator__max_iter": [1000], 
                  "estimator__tol": [1e-10]}
 
forward_scores, reverse_scores = [], []

def optimize_circuits(k):
    print("Assessing k={:02d} circuits".format(k))
    
    list_file = "../lists/lists_k{:02d}_oplen.csv".format(k)
    lists = pd.read_csv(list_file, index_col=None)
    
    circuit_file = "../circuits/circuits_k{:02d}.csv".format(k)
    circuits = pd.read_csv(circuit_file, index_col=None)
    
    function_features = pd.DataFrame(index=dtm_bin.index, columns=range(1, k+1))
    structure_features = pd.DataFrame(index=act_bin.index, columns=range(1, k+1))
    for i in range(1, k+1):
        functions = lists.loc[lists["CLUSTER"] == i, "TOKEN"]
        function_features[i] = dtm_bin[functions].sum(axis=1)
        structures = circuits.loc[circuits["CLUSTER"] == i, "STRUCTURE"]
        structure_features[i] = act_bin[structures].sum(axis=1)
    function_features = mean_thres(function_features)
    function_features = pd.DataFrame(binarize(function_features), 
                                     index=dtm_bin.index, columns=range(1, k+1))
    structure_features = pd.DataFrame(binarize(structure_features), 
                                     index=act_bin.index, columns=range(1, k+1))
    
    forward_cv = search_grid(function_features.loc[train], structure_features.loc[train], param_grid)
    forward_lr = forward_cv.best_estimator_
    forward_file = "fits/forward_k{:02d}_oplen.p".format(k)
    pickle.dump(forward_lr, open(forward_file, "wb"), protocol=2)
    forward_preds = forward_lr.predict(function_features.loc[val])
    forward_score = roc_auc_score(structure_features.loc[val], forward_preds)
    print("Forward score: {}".format(forward_score))
    forward_scores.append(forward_score)
    
    reverse_cv = search_grid(structure_features.loc[train], function_features.loc[train], param_grid)
    reverse_lr = reverse_cv.best_estimator_
    reverse_file = "fits/reverse_k{:02d}_oplen.p".format(k)
    pickle.dump(reverse_lr, open(reverse_file, "wb"), protocol=2)
    reverse_preds = reverse_lr.predict(structure_features.loc[val])
    reverse_score = roc_auc_score(function_features.loc[val], reverse_preds)
    print("Reverse score: {}".format(reverse_score))
    reverse_scores.append(reverse_score)

    out_df = pd.DataFrame({"k": k, 
                           "FORWARD_ROC_AUC": forward_scores, 
                           "REVERSE_ROC_AUC": reverse_scores}, 
                          columns=["k", "FORWARD_ROC_AUC", "REVERSE_ROC_AUC"])
    out_df.to_csv("../lists/lists_k{:02d}_opk.csv".format(k), index=None)
    