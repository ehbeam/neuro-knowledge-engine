#!/usr/bin/python

import os
import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler


def doc_mean_thres(df):
  doc_mean = df.mean()
  df_bin = 1.0 * (df.values > doc_mean.values)
  df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
  return df_bin


def load_doc_term_matrix(version=190325, binarize=True):
  dtm = pd.read_csv("../../../data/text/dtm_{}.csv.gz".format(version), compression="gzip", index_col=0)
  if binarize:
    dtm = doc_mean_thres(dtm)
  return dtm


def load_coordinates():
  atlas_labels = pd.read_csv("../../../data/brain/labels.csv")
  activations = pd.read_csv("../../../data/brain/coordinates.csv", index_col=0)
  activations = activations[atlas_labels["PREPROCESSED"]]
  return activations


def load_raw_domains(k):
  list_file = "../../lists/lists_k{:02d}.csv".format(k)
  lists = pd.read_csv(list_file, index_col=None)
  circuit_file = "../../circuits/circuits_k{:02d}.csv".format(k)
  circuits = pd.read_csv(circuit_file, index_col=None)
  return lists, circuits


def optimize_hyperparameters(param_list, train_set, val_set, max_iter=100):
  
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
    if preds_val.shape[1] == 2 and val_set[1].shape[1] == 1: # In case there is only one class
      preds_val = preds_val[:,1] # The second column is for the label 1
    score_val = roc_auc_score(val_set[1], preds_val, average="macro")
    print("\n   Validation Set ROC-AUC {:6.4f}\n".format(score_val))
    
    # Update outputs if this model is the best so far
    if score_val > op_score_val:
      print("   Best so far!\n")
      op_score_val = score_val
      op_fit = classifier

  return op_score_val


def optimize_list_len(k):

    # Load the data splits
    splits = {}
    for split in ["train", "validation"]:
      splits[split] = [int(pmid.strip()) for pmid in open("../../../data/splits/{}.txt".format(split), "r").readlines()]

    act_bin = load_coordinates()
    dtm_bin = load_doc_term_matrix(version=190325, binarize=True)

    lists, circuits = load_raw_domains(k)
    
    # Specify the hyperparameters for the randomized grid search
    param_grid = {"penalty": ["l1", "l2"],
                  "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  "fit_intercept": [True, False]}
    param_list = list(ParameterSampler(param_grid, n_iter=28, random_state=42))
    max_iter = 1000

    list_lens = range(5, 26)
    op_lists = pd.DataFrame()
    
    for circuit in range(1, k+1):

        print("-" * 100)
        print("Fitting models for domain {:02d}".format(circuit))
        forward_scores, reverse_scores = [], []
        structures = circuits.loc[circuits["CLUSTER"] == circuit, "STRUCTURE"]

        for list_len in list_lens:
            print("-" * 85)
            print("Fitting models for lists of length {:02d}".format(list_len))
            words = lists.loc[lists["CLUSTER"] == circuit, "TOKEN"][:list_len]

            # Optimize forward inference classifier 
            train_set_f = [dtm_bin.loc[splits["train"], words], act_bin.loc[splits["train"], structures]]
            val_set_f = [dtm_bin.loc[splits["validation"], words], act_bin.loc[splits["validation"], structures]]
            op_val_f = optimize_hyperparameters(param_list, train_set_f, val_set_f, max_iter=max_iter)
            forward_scores.append(op_val_f)

            # Optimize reverse inference classifier
            train_set_r = [act_bin.loc[splits["train"], structures], dtm_bin.loc[splits["train"], words]]
            val_set_r = [act_bin.loc[splits["validation"], structures], dtm_bin.loc[splits["validation"], words]]
            op_val_r = optimize_hyperparameters(param_list, train_set_r, val_set_r, max_iter=max_iter)
            reverse_scores.append(op_val_r)
        
        scores = [(forward_scores[i] + reverse_scores[i])/2.0 for i in range(len(forward_scores))]
        print("-" * 85)
        print("Mean ROC-AUC scores: {}".format(scores))
        op_len = list_lens[scores.index(max(scores))]
        print("-" * 100)
        print("\tCircuit {:02d} has {:02d} words".format(circuit, op_len))
        op_df = lists.loc[lists["CLUSTER"] == circuit][:op_len]
        op_df["ROC_AUC"] = max(scores)
        op_lists = op_lists.append(op_df)

    op_lists.to_csv("../../lists/lists_k{:02d}_oplen_logreg.csv".format(k), index=None)
