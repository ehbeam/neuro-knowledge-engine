#!/usr/bin/python

import pandas as pd
import numpy as np
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


def load_coordinates():
  atlas_labels = pd.read_csv("../../data/brain/labels.csv")
  activations = pd.read_csv("../../data/brain/coordinates.csv", index_col=0)
  activations = activations[atlas_labels["PREPROCESSED"]]
  return activations


def load_raw_domains(k):
  list_file = "../lists/lists_k{:02d}.csv".format(k)
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


def optimize_list_len(k):

    np.random.seed(42)

    act_bin = load_coordinates()
    dtm_bin = load_doc_term_matrix(version=190325, binarize=True)

    train = [int(pmid.strip()) for pmid in open("../../data/splits/train.txt")]
    val = [int(pmid.strip()) for pmid in open("../../data/splits/validation.txt")]

    lists, circuits = load_raw_domains(k)
    
    param_grid = {"estimator__hidden_layer_sizes": [(50,50)],
                  "estimator__activation": ["logistic"],
                  "estimator__solver": ["adam"],
                  "estimator__alpha": [1e-5],
                  "estimator__random_state": [42], 
                  "estimator__max_iter": [10]}

    list_lens = range(5, 26)
    op_lists = pd.DataFrame()
    
    for circuit in range(1, k+1):

        forward_scores, reverse_scores = [], []
        structures = circuits.loc[circuits["CLUSTER"] == circuit, "STRUCTURE"]

        for list_len in list_lens:
            print("-" * 80 + "\n" + "-" * 80)
            print("Fitting models for lists of length {:02d}".format(list_len))
            words = lists.loc[lists["CLUSTER"] == circuit, "TOKEN"][:list_len]

            # try:
            #     # forward_cv = search_grid(dtm_bin.loc[train, words], act_bin.loc[train, structures], 
            #     #                          param_grid, n_iter=1)
            #     # forward_clf = forward_cv.best_estimator_
            forward_clf = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(50,50), activation="logistic", solver="adam", alpha=1e-5, random_state=42, max_iter=10))
            forward_clf.fit(dtm_bin.loc[train, words], act_bin.loc[train, structures])
            forward_preds = forward_clf.predict(dtm_bin.loc[val, words])
            forward_scores.append(roc_auc_score(act_bin.loc[val, structures], forward_preds))
            # except:
            #     print("Unable to score forward models for length {:02d}".format(list_len))
            #     forward_scores.append(0.0)
            
            # try:
            #     # reverse_cv = search_grid(act_bin.loc[train, structures], dtm_bin.loc[train, words], 
            #     #                          param_grid, n_iter=1)
            #     # reverse_clf = reverse_cv.best_estimator_
            reverse_clf = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(50,50), activation="logistic", solver="adam", alpha=1e-5, random_state=42, max_iter=10))
            reverse_clf.fit(act_bin.loc[train, structures], dtm_bin.loc[train, words])
            reverse_preds = reverse_clf.predict(act_bin.loc[val, structures])
            reverse_scores.append(roc_auc_score(dtm_bin.loc[val, words], reverse_preds))
            # except:
            #     print("Unable to score reverse models for length {:02d}".format(list_len))
            #     reverse_scores.append(0.0)
        
        scores = [(forward_scores[i] + reverse_scores[i])/2.0 for i in range(len(forward_scores))]
        print("-" * 80)
        print("Mean ROC-AUC scores: {}".format(scores))
        op_len = list_lens[scores.index(max(scores))]
        print("-" * 100)
        print("\tCircuit {:02d} has {:02d} words".format(circuit, op_len))
        op_df = lists.loc[lists["CLUSTER"] == circuit][:op_len]
        op_df["ROC_AUC"] = max(scores)
        op_lists = op_lists.append(op_df)

    op_lists.to_csv("../lists/lists_k{:02d}_oplen.csv".format(k), index=None)
