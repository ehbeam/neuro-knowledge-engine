#!/usr/bin/python3

import os, pickle
import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.preprocessing import binarize
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


def load_lexicon(sources):
  lexicon = []
  for source in sources:
    file = "../../../lexicon/lexicon_{}.txt".format(source)
    lexicon += [token.strip() for token in open(file, "r").readlines()]
  return sorted(lexicon)


def load_coordinates():
  atlas_labels = pd.read_csv("../../../data/brain/labels.csv")
  activations = pd.read_csv("../../../data/brain/coordinates.csv", index_col=0)
  activations = activations[atlas_labels["PREPROCESSED"]]
  return activations


def load_domains(k):
  list_file = "../../lists/lists_k{:02d}_oplen_logreg.csv".format(k)
  lists = pd.read_csv(list_file, index_col=None)
  circuit_file = "../../circuits/circuits_k{:02d}.csv".format(k)
  circuits = pd.read_csv(circuit_file, index_col=None)
  return lists, circuits


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

 
def optimize_circuits(k, direction, max_iter=500):
  print("Assessing k={:02d} circuits".format(k))

  act_bin = load_coordinates()

  lexicon = load_lexicon(["cogneuro"])
  dtm_bin = load_doc_term_matrix(version=190325, binarize=True)
  lexicon = sorted(list(set(lexicon).intersection(dtm_bin.columns)))
  dtm_bin = dtm_bin[lexicon]

  lists, circuits = load_domains(k)

  function_features = pd.DataFrame(index=dtm_bin.index, columns=range(1, k+1))
  structure_features = pd.DataFrame(index=act_bin.index, columns=range(1, k+1))
  for i in range(1, k+1):
      functions = lists.loc[lists["CLUSTER"] == i, "TOKEN"]
      function_features[i] = dtm_bin[functions].sum(axis=1)
      structures = circuits.loc[circuits["CLUSTER"] == i, "STRUCTURE"]
      structure_features[i] = act_bin[structures].sum(axis=1)
  function_features = pd.DataFrame(doc_mean_thres(function_features),
                                   index=dtm_bin.index, columns=range(1, k+1))
  structure_features = pd.DataFrame(binarize(structure_features),
                                    index=act_bin.index, columns=range(1, k+1))

  # Load the data splits
  splits = {}
  for split in ["train", "validation"]:
    splits[split] = [int(pmid.strip()) for pmid in open("../../../data/splits/{}.txt".format(split), "r").readlines()]

  # Specify the hyperparameters for the randomized grid search
  param_grid = {"penalty": ["l1", "l2"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "fit_intercept": [True, False]}
  param_list = list(ParameterSampler(param_grid, n_iter=28, random_state=42))

  if direction == "forward":
    file = "fits/forward_k{:02d}_{}.p".format(k, direction)
    if not os.path.isfile(file):
      print("-" * 80 + "\nOptimizing forward model\n" + "-" * 80)
      train_set = [function_features.loc[splits["train"]], structure_features.loc[splits["train"]]]
      val_set = [function_features.loc[splits["validation"]], structure_features.loc[splits["validation"]]]
      op_fit = optimize_hyperparameters(param_list, train_set, val_set, max_iter=max_iter)
      pickle.dump(op_fit, open(file, "wb"), protocol=2)

  elif direction == "reverse":
    file = "fits/reverse_k{:02d}_{}.p".format(k, direction)
    if not os.path.isfile(file):
      print("-" * 80 + "\nOptimizing reverse model\n" + "-" * 80)
      train_set = [structure_features.loc[splits["train"]], function_features.loc[splits["train"]]]
      val_set = [structure_features.loc[splits["validation"]], function_features.loc[splits["validation"]]]
      op_fit = optimize_hyperparameters(param_list, train_set, val_set, max_iter=max_iter)
      pickle.dump(op_fit, open(file, "wb"), protocol=2)
      
