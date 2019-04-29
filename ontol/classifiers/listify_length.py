#!/usr/bin/python

import pandas as pd
import numpy as np
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

def optimize_list_len(k):

    np.random.seed(42)

    atlas_labels = pd.read_csv("../data/harvard-oxford.csv")
    act_bin = pd.read_csv("../data/dcm_0mm_thres0.csv", index_col=0)
    act_bin = act_bin[atlas_labels["PREPROCESSED"]]

    rdoc_version = 190124
    dtm = pd.read_csv("../data/dtm_{}.csv.gz".format(rdoc_version), compression="gzip", index_col=0)
    lexicon = [tkn.strip() for tkn in open("../data/lexicon_cogneuro_preproc.txt", "r") if tkn.strip() in list(dtm.columns)]
    rdoc_seeds = [seed for seed in set(pd.read_csv("../data/rdoc_{}/rdoc_seeds.csv".format(rdoc_version))["TOKEN"]) if seed in dtm.index]
    lexicon = sorted(list(set(lexicon + rdoc_seeds)))
    dtm = dtm[lexicon]

    dtm_bin = np.empty((dtm.shape[0], dtm.shape[1]))
    for i, (word, mean_docfreq) in enumerate(dtm.mean().iteritems()):
        dtm_bin[:,i] = 1 * (dtm[word] > mean_docfreq)
    dtm_bin = pd.DataFrame(dtm_bin, index=dtm.index, columns=dtm.columns)
    dtm_bin = mean_thres(dtm_bin)

    train = [int(pmid.strip()) for pmid in open("../data/splits/train.txt")]
    val = [int(pmid.strip()) for pmid in open("../data/splits/validation.txt")]

    circuit_file = "../circuits/circuits_k{:02d}.csv".format(k)
    circuits = pd.read_csv(circuit_file, index_col=None)
    
    list_file = "../lists/lists_k{:02d}.csv".format(k)
    lists = pd.read_csv(list_file, index_col=None)
    
    param_grid = {"estimator__penalty": ["l1", "l2"],
                  "estimator__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "estimator__fit_intercept": [True, False],
                  "estimator__solver": ["liblinear"],
                  "estimator__random_state": [42], 
                  "estimator__max_iter": [1000], 
                  "estimator__tol": [1e-10]}

    list_lens = range(5, 26)

    op_lists = pd.DataFrame()
    for circuit in range(1, k+1):

        forward_scores, reverse_scores = [], []
        structures = circuits.loc[circuits["CLUSTER"] == circuit, "STRUCTURE"]

        for list_len in list_lens:
            print("-" * 80 + "\n" + "-" * 80)
            print("Fitting models for lists of length {:02d}".format(list_len))
            words = lists.loc[lists["CLUSTER"] == circuit, "TOKEN"][:list_len]

            try:
                forward_cv = search_grid(dtm_bin.loc[train, words], act_bin.loc[train, structures], param_grid)
                forward_lr = forward_cv.best_estimator_
                forward_preds = forward_lr.predict(dtm_bin.loc[val, words])
                forward_scores.append(roc_auc_score(act_bin.loc[val, structures], forward_preds))
            except:
                forward_scores.append(0)
            
            try:
                reverse_cv = search_grid(act_bin.loc[train, structures], dtm_bin.loc[train, words], param_grid)
                reverse_lr = reverse_cv.best_estimator_
                reverse_preds = reverse_lr.predict(act_bin.loc[val, structures])
                reverse_scores.append(roc_auc_score(dtm_bin.loc[val, words], reverse_preds))
            except:
                reverse_scores.append(0)
        
        scores = [(forward_scores[i] + reverse_scores[i])/2.0 for i in range(len(forward_scores))]
        op_len = list_lens[scores.index(max(scores))]
        print("-" * 50)
        print("\tCircuit {:02d} has {:02d} words".format(circuit, op_len))
        op_df = lists.loc[lists["CLUSTER"] == circuit][:op_len]
        op_df["ROC_AUC"] = max(scores)
        op_lists = op_lists.append(op_df)

    path = "/scratch/PI/aetkin/ebeam/rdoc/rdoc-nlp"
    outfile = "{}/lists/lists_k{:02d}_oplen.csv".format(path, k)
    op_lists.to_csv(outfile, index=None)

