#!/usr/bin/python3

import math
import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(42)

import sys
sys.path.append("..")
import utilities


def plot_curves(metric, framework, direction, x, y, palette, 
				diag=True, opacity=0.5, font=utilities.arial, path="", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_prop = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the curves
	for i in range(len(x)):
		plt.plot(x[i], y[i], alpha=opacity, 
				 c=palette[i], linewidth=2)

	# Plot a diagonal line
	if diag:
		plt.plot([-1,2], [-1,2], linestyle="dashed", c="k", 
				 alpha=1, linewidth=2)

	plt.xlim([-0.05, 1])
	plt.ylim([-0.05, 1])
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)

	
	plt.xticks(fontproperties=font_prop)
	plt.yticks(fontproperties=font_prop)

	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	plt.savefig("{}figures/{}_{}_{}.png".format(path, metric, framework, direction), 
				bbox_inches="tight", dpi=250)
	if print_fig:
		plt.show()
	plt.close()


def compute_eval_metric(labels, preds, metric_function):
	import warnings
	metric_scores = []
	for i in range(labels.shape[1]):
		with warnings.catch_warnings(): # Ignore warning for classifiers that always predict 0
			warnings.filterwarnings("ignore") 
			metric_scores.append(metric_function(labels[:,i], preds[:,i], average="macro"))
	return metric_scores


def plot_eval_metric(metric, framework, direction, obs, boot, null_ci, fdr, 
					 palette, labels=[], dx=0.375, dxs=0.1, figsize=(11,4.5), 
					 ylim=[0.43,0.78], yticks=np.array(range(0,100,5))/100,
					 alpha=0.001, font=utilities.arial, print_fig=True, path=""):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_sm = font_manager.FontProperties(fname=font, size=8)
	font_lg = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0,0,1,1])

	# Plot the null distributions
	n_clf = len(obs)
	plt.fill_between(range(n_clf), null_ci["lower"], null_ci["upper"], 
					 alpha=0.2, color="gray")
	plt.plot(null_ci["mean"], 
			 linestyle="dashed", color="gray", linewidth=2)

	# Plot the bootstrap distributions
	for i in range(n_clf):
		plt.plot([i-dx, i+dx], [obs[i], obs[i]], 
				 c=palette[i], alpha=1, lw=2, zorder=-1)
		v = ax.violinplot(sorted(boot[i,:]), positions=[i], 
						  showmeans=False, showmedians=False, widths=0.85)
		for pc in v["bodies"]:
			pc.set_facecolor(palette[i])
			pc.set_edgecolor(palette[i])
			pc.set_linewidth(2)
			pc.set_alpha(0.5)
		for line in ["cmaxes", "cmins", "cbars"]:
			v[line].set_edgecolor("none")

		# Comparison test
		if fdr[i] < alpha:
			plt.text(i-dxs, max(boot[i,:]), "*", fontproperties=font_lg)

	ax.set_xticks(range(n_clf))
	ax.set_xticklabels(labels, rotation=60, ha="right")
	plt.xticks(fontproperties=font_sm)
	plt.yticks(yticks, fontproperties=font_lg)
	plt.xlim([-1, n_clf])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	n_iter = boot.shape[1]
	plt.savefig("{}figures/{}_{}_{}_{}iter.png".format(path, metric, framework, direction, n_iter), 
				dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()



def plot_loss(prefix, loss, xlab="", ylab="",
			  diag=True, alpha=0.5, color="gray", path=""):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_prop = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the loss curve
	plt.plot(range(len(loss)), loss, alpha=alpha, 
			 c=color, linewidth=2)

	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	plt.xticks(fontproperties=font_prop)
	plt.yticks(fontproperties=font_prop)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	plt.xlabel(xlab, fontproperties=font_prop)
	plt.ylabel(ylab, fontproperties=font_prop)
	ax.xaxis.set_label_coords(0.5, -0.165)
	ax.yaxis.set_label_coords(-0.275, 0.5)

	plt.savefig("{}figures/{}_loss.png".format(path, prefix), 
				bbox_inches="tight", dpi=250)
	plt.show()
	plt.close()


def compute_roc(labels, pred_probs):
	from sklearn.metrics import roc_curve
	fpr, tpr = [], []
	for i in range(labels.shape[1]):
		fpr_i, tpr_i, _ = roc_curve(labels[:,i], 
									pred_probs[:,i], pos_label=1)
		fpr.append(fpr_i)
		tpr.append(tpr_i)
	return fpr, tpr


def compute_prc(labels, pred_probs):
	from sklearn.metrics import precision_recall_curve
	precision, recall = [], []
	for i in range(labels.shape[1]):
		p_i, r_i, _ = precision_recall_curve(labels[:,i], 
											 pred_probs[:,i], pos_label=1)
		precision.append(p_i)
		recall.append(r_i)
	return precision, recall


def numpy2torch(data):
	inputs, labels = data
	inputs = Variable(torch.from_numpy(inputs.T).float())
	labels = Variable(torch.from_numpy(labels.T).float())
	return inputs, labels


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
	for k in range(0, int(num_complete_minibatches)):
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


def plot_framework_comparison(metric, direction, boot, n_iter=1000, font=utilities.arial,
							  dx=0.38, ylim=[0.4,0.65], yticks=[], path="", print_fig=True):
	

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_prop = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=(2.6,3.2))
	ax = fig.add_axes([0,0,1,1])

	i = 0
	labels = []
	for fw, dist in boot.items():
		labels.append(dist["name"])
		dist = dist[direction][metric]
		dist_avg = np.mean(dist, axis=0)
		macro_avg = np.mean(dist_avg)
		plt.plot([i-dx, i+dx], [macro_avg, macro_avg], 
				 c="gray", alpha=1, lw=2, zorder=-1)
		v = ax.violinplot(sorted(dist_avg), positions=[i], 
						  showmeans=False, showmedians=False, widths=0.85)
		for pc in v["bodies"]:
			pc.set_facecolor("gray")
			pc.set_edgecolor("gray")
			pc.set_linewidth(2)
			pc.set_alpha(0.5)
		for line in ["cmaxes", "cmins", "cbars"]:
			v[line].set_edgecolor("none")
		i += 1

	ax.set_xticks(range(len(boot.keys())))
	ax.set_xticklabels(labels, rotation=60, ha="right")
	plt.xticks(fontproperties=font_prop)
	plt.yticks(yticks, fontproperties=font_prop)
	plt.xlim([-0.75, len(boot.keys())-0.25])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	plt.savefig("{}figures/{}_{}_{}iter.png".format(path, metric, direction, n_iter), 
				dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()

