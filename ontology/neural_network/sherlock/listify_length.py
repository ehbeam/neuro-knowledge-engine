#!/usr/bin/python

import os, math
import pandas as pd
import numpy as np
np.random.seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(42)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler


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


def numpy2torch(data):
  inputs, labels = data
  inputs = Variable(torch.from_numpy(inputs.T).float())
  labels = Variable(torch.from_numpy(labels.T).float())
  return inputs, labels


def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()


class Net(nn.Module):
  def __init__(self, n_input=0, n_output=0, n_hid=100, p_dropout=0.5):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_input, n_hid)
    self.bn1 = nn.BatchNorm1d(n_hid)
    self.dropout1 = nn.Dropout(p=p_dropout)
    self.fc2 = nn.Linear(n_hid, n_hid)
    self.bn2 = nn.BatchNorm1d(n_hid)
    self.dropout2 = nn.Dropout(p=p_dropout)
    self.fc3 = nn.Linear(n_hid, n_hid)
    self.bn3 = nn.BatchNorm1d(n_hid)
    self.dropout3 = nn.Dropout(p=p_dropout)
    self.fc4 = nn.Linear(n_hid, n_hid)
    self.bn4 = nn.BatchNorm1d(n_hid)
    self.dropout4 = nn.Dropout(p=p_dropout)
    self.fc5 = nn.Linear(n_hid, n_hid)
    self.bn5 = nn.BatchNorm1d(n_hid)
    self.dropout5 = nn.Dropout(p=p_dropout)
    self.fc6 = nn.Linear(n_hid, n_hid)
    self.bn6 = nn.BatchNorm1d(n_hid)
    self.dropout6 = nn.Dropout(p=p_dropout)
    self.fc7 = nn.Linear(n_hid, n_hid)
    self.bn7 = nn.BatchNorm1d(n_hid)
    self.dropout7 = nn.Dropout(p=p_dropout)
    self.fc8 = nn.Linear(n_hid, n_output)
    
    # Xavier initialization for weights
    for fc in [self.fc1, self.fc2, self.fc3, self.fc4,
           self.fc5, self.fc6, self.fc7, self.fc8]:
      nn.init.xavier_uniform_(fc.weight)

  def forward(self, x):
    x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
    x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
    x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
    x = self.dropout4(F.relu(self.bn4(self.fc4(x))))
    x = self.dropout5(F.relu(self.bn5(self.fc5(x))))
    x = self.dropout6(F.relu(self.bn6(self.fc6(x))))
    x = self.dropout7(F.relu(self.bn7(self.fc7(x))))
    x = torch.sigmoid(self.fc8(x))
    return x
    

def optimize_hyperparameters(param_list, train_set, val_set, n_epochs=100):
  
  criterion = F.binary_cross_entropy
  inputs_val, labels_val = numpy2torch(val_set[0])
  op_idx, op_params, op_score_val, op_state_dict, op_loss = 0, 0, 0, 0, 0
  
  for params in param_list:
    
    print("-" * 75)
    print("   ".join(["{} {:6.5f}".format(k.upper(), v) for k, v in params.items()]))
    print("-" * 75 + "\n")
    
    # Initialize variables for this set of parameters
    n_input = train_set[0][0].shape[0]
    n_output = train_set[0][1].shape[0]
    net = Net(n_input=n_input, n_output=n_output, 
      n_hid=params["n_hid"], p_dropout=params["p_dropout"])
    optimizer = optim.Adam(net.parameters(),
                 lr=params["lr"], weight_decay=params["weight_decay"])
    net.apply(reset_weights)
    running_loss = []

    # Loop over the dataset multiple times
    for epoch in range(n_epochs): 
      for data in train_set:

        # Get the inputs
        inputs, labels = numpy2torch(data)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
      
      # Update the running loss
      running_loss += [loss.item()]
      if epoch % (n_epochs/5) == (n_epochs/5) - 1:
        print("   Epoch {:3d}\tLoss {:6.6f}".format(epoch + 1, running_loss[-1] / 100)) 
    
    # Evaluate on the validation set
    with torch.no_grad():
      preds_val = net.eval()(inputs_val).float()
    score_val = roc_auc_score(labels_val, preds_val, average="macro")
    print("\n   Validation Set ROC-AUC {:6.4f}\n".format(score_val))
    
    # Update outputs if this model is the best so far
    if score_val > op_score_val:
      print("   Best so far!\n")
      op_score_val = score_val
      op_state_dict = net.state_dict()
      op_params = params
      op_loss = running_loss

  return op_score_val


def load_mini_batches(X, Y, split, mini_batch_size=64, seed=0, reshape_labels=False):
  
  np.random.seed(seed)      
  m = len(split) # Number of training examples
  mini_batches = []

  # Split the data
  X = X.loc[split].T.values
  Y = Y.loc[split].T.values
    
  # Shuffle (X, Y)
  permutation = list(np.random.permutation(m))
  shuffled_X = X[:, permutation]
  shuffled_Y = Y[:, permutation]
  if reshape_labels:
    shuffled_Y = shuffled_Y.reshape((1,m))

  # Partition (shuffled_X, shuffled_Y), except the end case
  num_complete_minibatches = math.floor(m / mini_batch_size) # Mumber of mini batches of size mini_batch_size in your partitionning
  for k in range(0, num_complete_minibatches):
    mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
    mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
  
  # Handle the end case (last mini-batch < mini_batch_size)
  if m % mini_batch_size != 0:
    mini_batch_X = shuffled_X[:, -(m % mini_batch_size):]
    mini_batch_Y = shuffled_Y[:, -(m % mini_batch_size):]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
  
  return mini_batches


def optimize_list_len(k):

    # Load the data splits
    splits = {}
    for split in ["train", "validation"]:
      splits[split] = [int(pmid.strip()) for pmid in open("../../data/splits/{}.txt".format(split), "r").readlines()]

    act_bin = load_coordinates()
    dtm_bin = load_doc_term_matrix(version=190325, binarize=True)

    lists, circuits = load_raw_domains(k)
    
    # Specify the hyperparameters for the randomized grid search
    param_grid = {"lr": [0.001],
                  "weight_decay": [0.001],
                  "n_hid": [100],
                  "p_dropout": [0.1]}
    param_list = list(ParameterSampler(param_grid, n_iter=1, random_state=42))
    batch_size = 1024
    n_epochs = 100

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
            train_set_f = load_mini_batches(dtm_bin[words], act_bin[structures], splits["train"], mini_batch_size=batch_size, seed=42)
            val_set_f = load_mini_batches(dtm_bin[words], act_bin[structures], splits["validation"], mini_batch_size=len(splits["validation"]), seed=42)
            op_val_f = optimize_hyperparameters(param_list, train_set_f, val_set_f, n_epochs=n_epochs)
            forward_scores.append(op_val_f)

            # Optimize reverse inference classifier
            train_set_r = load_mini_batches(act_bin[structures], dtm_bin[words], splits["train"], mini_batch_size=batch_size, seed=42)
            val_set_r = load_mini_batches(act_bin[structures], dtm_bin[words], splits["validation"], mini_batch_size=len(splits["validation"]), seed=42)
            op_val_r = optimize_hyperparameters(param_list, train_set_r, val_set_r, n_epochs=n_epochs)
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

    op_lists.to_csv("../lists/lists_k{:02d}_oplen_nn.csv".format(k), index=None)
