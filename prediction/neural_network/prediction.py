#!/usr/bin/python3

import os, math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(42)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler

import sys
sys.path.append("../..")
import utilities


def numpy2torch(data):

	inputs, labels = data
	inputs = Variable(torch.from_numpy(inputs.T).float())
	labels = Variable(torch.from_numpy(labels.T).float())
	
	return inputs, labels


def load_mini_batches(X, Y, split, mini_batch_size=64, seed=42, reshape_labels=False):

	# Adapted from https://www.coursera.org/learn/deep-neural-network

	import math
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
	num_complete_minibatches = math.floor(m / mini_batch_size) # Number of mini batches of size mini_batch_size
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
		print("   ".join(["{} {}".format(k.upper(), v) for k, v in params.items()]))
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
			if len(param_list) > 1:
				print("   Best so far!\n")
			op_score_val = score_val
			op_state_dict = net.state_dict()
			op_params = params
			op_loss = running_loss

	return op_state_dict, op_params, op_loss


def train_classifier(framework, direction, suffix="", clf="", dtm_version=190325, 
					 opt_epochs=500, train_epochs=1000, use_hyperparams=False):

	# Load the data splits
	splits = {}
	for split in ["train", "validation"]:
		splits[split] = [int(pmid.strip()) for pmid in open("../../data/splits/{}.txt".format(split), "r").readlines()]

	# Load the activation coordinate and text data
	act_bin = utilities.load_coordinates(path="../../data")
	dtm_bin = utilities.load_doc_term_matrix(version=dtm_version, binarize=True, path="../../data")
	
	# Score the texts using the framework
	lists, circuits = utilities.load_framework(framework, suffix=suffix, clf=clf, path="../../ontology")
	scores = utilities.score_lists(lists, dtm_bin)
		
	# If hyperparameters have already been optimizd, use them
	param_file = "../data/params_{}_{}_{}epochs.csv".format(framework, direction, opt_epochs)
	if use_hyperparams:
		params = pd.read_csv(param_file, header=None, index_col=0)
		param_grid = {"lr": [float(params.loc["lr"])],
			  		  "weight_decay": [float(params.loc["weight_decay"])],
			  		  "n_hid": [int(params.loc["n_hid"])],
			  		  "p_dropout": [float(params.loc["p_dropout"])]}
		param_list = list(ParameterSampler(param_grid, n_iter=1, random_state=42))
		n_epochs = train_epochs

	# Otherwise, specify hyperparameters for a randomized grid search
	elif not use_hyperparams:
		param_grid = {"lr": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
			  		  "weight_decay": [0.00001, 0.0001, 0.001, 0.01, 0.1],
			  		  "n_hid": [25, 50, 75, 100, 125, 150],
			  		  "p_dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
		param_list = list(ParameterSampler(param_grid, n_iter=100, random_state=42))
		n_epochs = opt_epochs
	batch_size = 1024
	
	# Split the train set into batches and load the validation set as a batch
	if direction == "forward":
		train_set = load_mini_batches(scores, act_bin, splits["train"], mini_batch_size=batch_size, seed=42)
		val_set = load_mini_batches(scores, act_bin, splits["validation"], mini_batch_size=len(splits["validation"]), seed=42)
	
	elif direction == "reverse":
		train_set = load_mini_batches(act_bin, scores, splits["train"], mini_batch_size=batch_size, seed=42)
		val_set = load_mini_batches(act_bin, scores, splits["validation"], mini_batch_size=len(splits["validation"]), seed=42)

	# Search for the optimal hyperparameter combination
	op_state_dict, op_params, op_loss = optimize_hyperparameters(param_list, train_set, val_set, n_epochs=n_epochs)

	# Export the trained neural network
	fit_file = "../fits/{}_{}_{}epochs.pt".format(framework, direction, n_epochs)
	torch.save(op_state_dict, fit_file)
	
	# Export the hyperparameters
	with open(param_file, "w+") as file:
		file.write("\n".join(["{},{}".format(param, val) for param, val in op_params.items()]))

	# Export the loss over epochs
	loss_file = "../data/loss_{}_{}_{}epochs.csv".format(framework, direction, n_epochs)
	pd.DataFrame(op_loss, index=None, columns=["LOSS"]).to_csv(loss_file)
