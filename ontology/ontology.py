#!/usr/bin/python3

import os
import pandas as pd
import numpy as np
np.random.seed(42)

import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
torch.manual_seed(42)

import sys
sys.path.append("..")
from utilities import *

from matplotlib import font_manager, rcParams
font_md = font_manager.FontProperties(fname=arial, size=20)
font_lg = font_manager.FontProperties(fname=arial, size=22)
rcParams["axes.linewidth"] = 1.5

from scipy.spatial.distance import cdist


def load_ontology(k, path="", suffix=""):
	list_file = "{}lists/lists_k{:02d}_oplen{}.csv".format(path, k, suffix)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "{}circuits/circuits_k{:02d}.csv".format(path, k)
	circuits = pd.read_csv(circuit_file, index_col=None)
	return lists, circuits


def observed_over_expected(df):
	# From https://github.com/cgpotts/cs224u/blob/master/vsm.py
	col_totals = df.sum(axis=0)
	total = col_totals.sum()
	row_totals = df.sum(axis=1)
	expected = np.outer(row_totals, col_totals) / total
	oe = df / expected
	return oe


def pmi(df, positive=True):
	# From https://github.com/cgpotts/cs224u/blob/master/vsm.py
	df = observed_over_expected(df)
	with np.errstate(divide="ignore"):
		df = np.log(df)
	df[np.isinf(df)] = 0.0  # log(0) = 0
	if positive:
		df[df < 0] = 0.0
	return df


def load_stm(act_bin, dtm_bin):
	stm = np.dot(act_bin.transpose(), dtm_bin)
	stm = pd.DataFrame(stm, columns=dtm_bin.columns, index=act_bin.columns)
	stm = pmi(stm, positive=False)
	stm = stm.dropna(axis=1, how="all") # Drop terms with no co-occurrences
	return stm


def cluster_structures(k, stm, structures):
	kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=42)
	kmeans.fit(stm)
	clust = pd.DataFrame({"STRUCTURE": structures, 
						  "CLUSTER": [l+1 for l in list(kmeans.labels_)]})
	clust = clust.sort_values(["CLUSTER", "STRUCTURE"])
	return clust


def assign_functions(clust, splits, act_bin, dtm_bin, list_lens=range(5,26)):

	from scipy.stats import pointbiserialr

	lists = pd.DataFrame()
	for i in range(k):
		structures = list(clust.loc[clust["CLUSTER"] == i+1, "STRUCTURE"])
		centroid = np.mean(act_bin.loc[splits["train"], structures], axis=1)
		R = pd.Series([pointbiserialr(dtm_bin.loc[splits["train"], word], centroid)[0] 
					   for word in lexicon], index=lexicon)
		R = R[R > 0].sort_values(ascending=False)[:max(list_lens)]
		R = pd.DataFrame({"CLUSTER": [i+1 for l in range(max(list_lens))], 
						  "TOKEN": R.index, "R": R.values})
		lists = lists.append(R)
	return lists


def compute_cooccurrences(activations, scores):
	X = np.matmul(activations.values.T, scores.values)
	X = pmi(X, positive=True)
	X = pd.DataFrame(X, columns=scores.columns, index=activations.columns)
	X = X.dropna(axis=1, how="any")
	X = X.loc[:, (X != 0).any(axis=0)]
	return X


def compute_cooccurrences_null(activations, scores, n_iter=1000, verbose=False):
	np.random.seed(42)
	X_null = np.empty((activations.shape[1], scores.shape[1], n_iter))
	act_mat = activations.values.T
	scores_mat = scores.values
	n_docs = len(scores)
	for n in range(n_iter):
		null = np.random.choice(range(n_docs), size=n_docs, replace=False)
		X = np.matmul(act_mat, scores_mat[null,:])
		X = pmi(X, positive=True)
		X_null[:,:,n] = X
		if verbose:
			if n % (n_iter/10) == 0:
				print("Iteration {}".format(n))
	return X_null


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


def compute_eval_scores(scoring_function, directions, circuit_counts, features, fits, ids):
	
	eval_scores = {direction: np.zeros((len(circuit_counts))) for direction in directions}
	
	for i, k in enumerate(circuit_counts):
		
		with torch.no_grad():
			data_set = load_mini_batches(features[k]["function"].loc[ids], features[k]["structure"].loc[ids], 
										ids, mini_batch_size=len(ids), seed=42)
			function_features, structure_features = numpy2torch(data_set[0])
			y_pred_for = fits["forward"][k].eval()(function_features).float()
			y_true_for = structure_features
			y_pred_rev = fits["reverse"][k].eval()(structure_features).float()
			y_true_rev = function_features

		score_for = scoring_function(y_true_for, y_pred_for, average="macro")
		eval_scores["forward"][i] = score_for
		score_rev = scoring_function(y_true_rev, y_pred_rev, average="macro")
		eval_scores["reverse"][i] = score_rev

	eval_scores["mean"] = np.mean([eval_scores["forward"], eval_scores["reverse"]], axis=0)
	return eval_scores


def load_eval_data(features, k, ids):
	with torch.no_grad():
		data_set = load_mini_batches(features[k]["function"].loc[ids], features[k]["structure"].loc[ids], 
									ids, mini_batch_size=len(ids), seed=42)
		function_features, structure_features = numpy2torch(data_set[0])
	return function_features, structure_features


def load_eval_preds(clf, features):
	with torch.no_grad():
		preds = clf.eval()(features).float()
	return preds


def compute_eval_boot(scoring_function, directions, circuit_counts, 
					  features, fits, ids, n_iter=1000, verbose=True, path=""):
	eval_boot = {direction: np.zeros((len(circuit_counts), n_iter)) for direction in directions}

	file_for = "{}data/circuits_forward_boot_{}iter.csv".format(path, n_iter)
	if not os.path.exists(file_for):
		print("Bootstrap for N Domains | Forward")
		for i, k in enumerate(circuit_counts):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			function_features, structure_features = load_eval_data(features, k, ids)
			y_pred_for = load_eval_preds(fits["forward"][k], function_features)
			y_true_for = structure_features
			for n in range(n_iter):
				boot = np.random.choice(range(len(ids)), size=len(ids), replace=True)
				score_for = scoring_function(y_true_for[boot,:], y_pred_for[boot,:], average="macro")
				eval_boot["forward"][i,n] = score_for
		pd.DataFrame(eval_boot["forward"]).to_csv(file_for)
	elif os.path.exists(file_for):
		eval_boot["forward"] = pd.read_csv(file_for, index_col=0, header=0).values

	file_rev = "{}data/circuits_reverse_boot_{}iter.csv".format(path, n_iter)
	if not os.path.exists(file_rev):
		print("Bootstrap for N Domains | Reverse")
		for i, k in enumerate(circuit_counts):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			with torch.no_grad():
				function_features, structure_features = load_eval_data(features, k, ids)
				y_pred_rev = load_eval_preds(fits["reverse"][k], structure_features)
				y_true_rev = function_features
			for n in range(n_iter):
				boot = np.random.choice(range(len(ids)), size=len(ids), replace=True)
				score_rev = scoring_function(y_true_rev[boot,:], y_pred_rev[boot,:], average="macro")
				eval_boot["reverse"][i,n] = score_rev
		pd.DataFrame(eval_boot["reverse"]).to_csv(file_rev)
		print("")
	elif os.path.exists(file_rev):
		eval_boot["reverse"] = pd.read_csv(file_rev, index_col=0, header=0).values

	eval_boot["mean"] = np.mean([eval_boot["forward"], eval_boot["reverse"]], axis=0)
	return eval_boot


def compute_eval_null(scoring_function, directions, circuit_counts,
					  features, fits, ids, n_iter=1000, verbose=True, path=""):
	eval_null = {direction: np.zeros((len(circuit_counts), n_iter)) for direction in directions}
	
	file_for = "{}data/circuits_forward_null_{}iter.csv".format(path, n_iter)
	if not os.path.exists(file_for):
		print("Permutation for N Domains | Forward")
		for i, k in enumerate(circuit_counts):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			with torch.no_grad():
				function_features, structure_features = load_eval_data(features, k, ids)
				y_pred_for = load_eval_preds(fits["forward"][k], function_features)
				y_true_for = structure_features
			for n in range(n_iter):
				null = np.random.choice(range(len(ids)), size=len(ids), replace=False)
				score_for = scoring_function(y_true_for[null,:], y_pred_for, average="macro")
				eval_null["forward"][i,n] = score_for
		pd.DataFrame(eval_null["forward"]).to_csv(file_for)
	elif os.path.exists(file_for):
		eval_null["forward"] = pd.read_csv(file_for, index_col=0, header=0).values

	file_rev = "{}data/circuits_reverse_null_{}iter.csv".format(path, n_iter)
	if not os.path.exists(file_rev):
		print("Permutation for N Domains | Reverse")
		for i, k in enumerate(circuit_counts):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			with torch.no_grad():
				function_features, structure_features = load_eval_data(features, k, ids)
				y_pred_rev = load_eval_preds(fits["reverse"][k], structure_features)
				y_true_rev = function_features
			for n in range(n_iter):
				null = np.random.choice(range(len(ids)), size=len(ids), replace=False)
				score_rev = scoring_function(y_true_rev[null,:], y_pred_rev, average="macro")
				eval_null["reverse"][i,n] = score_rev
		pd.DataFrame(eval_null["reverse"]).to_csv(file_rev)
		print("")
	elif os.path.exists(file_for):
		eval_null["reverse"] = pd.read_csv(file_for, index_col=0, header=0).values

	eval_null["mean"] = np.mean([eval_null["forward"], eval_null["reverse"]], axis=0)
	return eval_null


def plot_scores(scores, direction, circuit_counts, boot, null, label, shape="o", op_k=6, 
				ylim=[0,1], yticks=[], interval=0.999, font=arial, path="", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_axes([0,0,1,1])
	font_prop = font_manager.FontProperties(fname=font, size=20)

	# Plot distribution
	null_lower, null_upper = load_confidence_interval(null, direction, circuit_counts, interval=interval)
	plt.fill_between(circuit_counts, null_lower, null_upper, 
					 alpha=0.2, color="gray")
	plt.plot(circuit_counts, np.mean(null[direction][:len(circuit_counts)], axis=1), 
			 linestyle="dashed", c="k", alpha=1, linewidth=2)

	# Plot bootstrap distribution
	n_iter = float(boot[direction].shape[1])
	boot_lower, boot_upper = load_confidence_interval(boot, direction, circuit_counts, interval=interval)
	for j, k in enumerate(circuit_counts):
		plt.plot([k, k], [boot_lower[j], boot_upper[j]], c="k", 
				 linewidth=9, alpha=0.15)

	# Plot observed values
	plt.scatter(circuit_counts, scores[direction][:len(circuit_counts)], color="black", 
				marker=shape, zorder=1000, s=45, alpha=0.3)
	xp = np.linspace(circuit_counts[0], circuit_counts[-1], 100)
	idx = np.isfinite(scores[direction][:len(circuit_counts)])
	p = np.poly1d(np.polyfit(np.array(circuit_counts)[idx], scores[direction][:len(circuit_counts)][idx], 2))
	plt.plot(xp, p(xp), zorder=0, linewidth=2, 
			 color="black", alpha=1)

	# Plot selected value
	op_idx = list(circuit_counts).index(op_k)
	plt.scatter(circuit_counts[op_idx]+0.03, scores[direction][:len(circuit_counts)][op_idx]-0.00015, 
				linewidth=2.5, edgecolor="black", color="none", 
				marker=shape, s=70, zorder=100)

	plt.xlim([0,max(circuit_counts)+1])
	plt.ylim(ylim)
	plt.xticks(fontproperties=font_prop)
	plt.yticks(yticks,fontproperties=font_prop)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)

	n_iter = boot[direction].shape[1]
	plt.savefig("{}figures/data-driven_{}_{}_{}iter.png".format(path, label, direction, n_iter), 
				dpi=250, bbox_inches="tight")
	if print_fig:
	   plt.show()
	plt.close()


def load_confidence_interval(distribution, direction, circuit_counts, interval=0.999):
	n_iter = float(distribution[direction].shape[1])
	lower = [sorted(distribution[direction][k,:])[int(n_iter*(1.0-interval))] for k in range(len(circuit_counts))]
	upper = [sorted(distribution[direction][k,:])[int(n_iter*interval)] for k in range(len(circuit_counts))]
	return lower, upper


def term_degree_centrality(i, lists, circuits, dtm, ids, reweight=False):
	terms = list(set(lists.loc[lists["CLUSTER"] == i, "TOKEN"]))
	ttm = pd.DataFrame(np.matmul(dtm.loc[ids, terms].T, dtm.loc[ids, terms]), 
					   index=terms, columns=terms)
	adj = pd.DataFrame(0, index=terms, columns=terms)
	for term_i in terms:
		for term_j in terms:
			adj.loc[term_i, term_j] = ttm.loc[term_i, term_j]
	degrees = adj.sum(axis=1)
	degrees = degrees.loc[terms]
	degrees = degrees.sort_values(ascending=False)
	return degrees


def plot_wordclouds(framework, domains, lists, dtm, path="", 
					font_path=arial, print_fig=True, width=550):
	
	from wordcloud import WordCloud
	import matplotlib.pyplot as plt

	for i, dom in enumerate(domains):
		
		def color_func(word, font_size, position, orientation, 
					   random_state=None, idx=0, **kwargs):
			return palettes[framework][i]

		tkns = lists.loc[lists["DOMAIN"] == dom, "TOKEN"]
		freq = dtm[tkns].sum().values
		tkns = [t.replace("_", " ") for t in tkns]

		cloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGB", 
						  max_font_size=100, prefer_horizontal=1, scale=20, margin=3,
						  width=width, height=15*len(tkns)+550, font_path=arial, 
						  random_state=42).generate_from_frequencies(zip(tkns, freq))

		fig = plt.figure()
		plt.axis("off")
		plt.imshow(cloud.recolor(color_func=color_func, random_state=42))
		file_name = "{}figures/lists/{}_wordcloud_{}.png".format(path, framework, dom)
		plt.savefig(file_name, 
					dpi=800, bbox_inches="tight")
		transparent_background(file_name)
		if print_fig:
			print(dom)
			plt.show()
		plt.close()


def load_rdoc_lists(lexicon, vsm, seed_df, n_thres=25, verbose=False):
	
	from collections import OrderedDict
	from scipy.spatial.distance import cdist

	lists = pd.DataFrame()
	labels = list(OrderedDict.fromkeys(seed_df["DOMAIN"]))
	for l, label in enumerate(labels):
		dom_df = seed_df.loc[seed_df["DOMAIN"] == label]
		tokens = list(dom_df["TOKEN"])
		centroid = np.mean(vsm.loc[tokens]).values.reshape(1, -1)
		dists = cdist(vsm.loc[lexicon], centroid, metric="cosine")
		dists = pd.Series(dists.reshape(-1), index=lexicon).sort_values()
		dists = [(w, d) for w, d in dists.iteritems()][:n_thres]
		
		if verbose:
			if len(dists) == 0:
				print("No tokens assigned to {}".format(label))
			if len(dists) < n_thres:
				print("{} tokens assigned to {}".format(len(dists), label))
		
		for w, d in dists:
			dic = {"ORDER": [l],
				   "DOMAIN": [label],
				   "TOKEN": [w],
				   "SOURCE": ["RDoC" if w in tokens else "Lexicon"],
				   "DISTANCE": [d]}
			lists = lists.append(pd.DataFrame(dic))
	
	lists = lists[["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]]
	lists = lists.sort_values(["ORDER", "DISTANCE"])
	return lists


def load_dsm_lists(lexicon, vsm, seed_df, unique, n_thres=25, verbose=False):
	
	lists = pd.DataFrame()
	for label in set(seed_df["DOMAIN"]):
		dom_df = seed_df.loc[seed_df["DOMAIN"] == label]
		tokens = set(dom_df["TOKEN"]).intersection(vsm.index)
		required = tokens.intersection(unique)
		forbidden = set(unique).difference(tokens)
		centroid = np.mean(vsm.loc[tokens]).values.reshape(1, -1)
		dists = cdist(vsm.loc[lexicon], centroid, metric="cosine")
		dists = pd.Series(dists.reshape(-1), index=lexicon).sort_values()
		dists = dists[dists < 0.5]
		dists = [(w, d) for w, d in dists.iteritems() 
				 if (w in required) or (w not in forbidden)][:n_thres]
		
		if verbose:
			if len(dists) == 0:
				print("No tokens assigned to {}".format(label))
		
		for w, d in dists:
			dic = {"ORDER": [list(dom_df["ORDER"])[0] + 1],
				   "DOMAIN": [label],
				   "TOKEN": [w],
				   "SOURCE": ["DSM" if w in tokens else "Lexicon"],
				   "DISTANCE": [d]}
			lists = lists.append(pd.DataFrame(dic))
	
	lists = lists[["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]]
	lists = lists.sort_values(["ORDER", "DISTANCE"])
	return lists


def load_optimized_lists(doms, lists, list_lens, seed_df, vsm):

	from scipy.spatial.distance import cosine

	ops = []
	op_df = pd.DataFrame(index=doms, columns=list_lens)
	for dom in doms:
		seed_tkns = seed_df.loc[seed_df["DOMAIN"] == dom, "TOKEN"]
		seed_centroid = np.mean(vsm.loc[seed_tkns])
		for list_len in list_lens:
			len_tkns = lists.loc[lists["DOMAIN"] == dom, "TOKEN"][:list_len]
			len_centroid = np.mean(vsm.loc[len_tkns])
			op_df.loc[dom, list_len] = 1 - cosine(seed_centroid, len_centroid)
		sims = list(op_df.loc[dom])
		idx = sims.index(max(sims))
		ops.append(np.array(list_lens)[idx])
	op_df["OPTIMAL"] = ops
	
	return op_df


def update_lists(doms, op_df, lists, framework):
	columns = ["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]
	new = pd.DataFrame(columns=columns)
	for order, dom in enumerate(doms):
		list_len = op_df.loc[dom, "OPTIMAL"]
		dom_df = lists.loc[lists["DOMAIN"] == dom][:list_len]
		new = new.append(dom_df)
	new.to_csv("ontology/lists/lists_{}_opsim.csv".format(framework), index=None)
	return new

def compute_centroid(df, labels, vsm, level="DOMAIN"):
	centroids = []
	for i, label in enumerate(labels):
		tkns = df.loc[df[level] == label, "TOKEN"]
		tkns = [tkn for tkn in tkns if tkn in vsm.index]
		centroids.append(np.mean(vsm.loc[tkns]))
	return np.array(centroids)


def compute_sims_sample(df, seed_centroid, vsm):
	idx = np.random.choice(range(vsm.shape[1]), size=vsm.shape[1], replace=True)
	sims = cdist(seed_centroid[:,idx], df[:,idx], "cosine")
	return np.diagonal(sims)


def compute_sims_shuffle(df, seed_centroid, vsm):
	idx_i = np.random.choice(range(vsm.shape[1]), size=vsm.shape[1], replace=False)
	idx_j = np.random.choice(range(vsm.shape[1]), size=vsm.shape[1], replace=False)
	sims = cdist(seed_centroid[:,idx_i], df[:,idx_j], "cosine")
	return np.diagonal(sims)


def compute_sims(df, seed_centroid, labels, vsm, level="DOMAIN"):
	centroids = compute_centroid(df, labels, vsm, level=level)
	sims = cdist(seed_centroid, centroids, "cosine")
	return np.diagonal(sims) 


def report_significance(pvals, labels, alphas=[0.01, 0.001, 0.0001]):
	for p, lab in zip(pvals, labels):
		stars = "".join(["*" for alpha in alphas if p < alpha])
		print("{:22s} p={:6.6f} {}".format(lab, p, stars))
		

def plot_violin(ax, boot, obs, labs, colors, alphas, font_xlg, fdrs=[], sig=False):

	import matplotlib.pyplot as plt

	for i, lab in enumerate(labs):
		data = sorted(boot[i,:])
		v = ax.violinplot(data, positions=[i], widths=0.8, 
						  showmeans=False, showmedians=False)
		for pc in v["bodies"]:
			pc.set_facecolor(colors[i])
			pc.set_edgecolor(colors[i])
			pc.set_linewidth(0.75)
			pc.set_alpha(0.6)
		for line in ["cmaxes", "cmins", "cbars"]:
			v[line].set_edgecolor("none")
		plt.plot([i-0.355, i+0.36], [obs[i], obs[i]], 
				 c=colors[i], alpha=1, lw=2)
		if sig and i < len(fdrs):
			fdr = fdrs[i]
			for alpha, y in zip(alphas, [0, 0.0425, 0.085]):
				if fdr < alpha:
					plt.text(i-0.1325, max(data) + y, "*", 
							 fontproperties=font_xlg)


def plot_rdoc_similarity(doms, new_sim_null, lower, upper, mccoy_sim_boot, mccoy_sim_obs,
						 new_sim_boot, new_sim_obs, fdrs_dif, alphas, font=arial):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams
	
	font_lg = font_manager.FontProperties(fname=font, size=20)
	font_xlg = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=(2.9, 4.5))
	ax = fig.add_axes([0,0,1,1])
	plt.plot(range(len(doms)), new_sim_null.mean(axis=1),
			 "gray", linestyle="dashed", linewidth=2)
	plt.fill_between(range(len(doms)), lower, y2=upper, 
					 color="gray", alpha=0.2)
	plot_violin(ax, mccoy_sim_boot, mccoy_sim_obs, doms[:-1], ["k"]*5, alphas, font_xlg), 
	plot_violin(ax, new_sim_boot, new_sim_obs, doms, palettes["rdoc"], alphas, font_xlg,
				fdrs=fdrs_dif, sig=True)
	plt.xlim([-0.75, len(doms)-0.5])
	plt.ylim([-0.3, 1.1])
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	ax.set_xticks(range(len(doms)))
	ax.set_xticklabels([])
	plt.yticks(fontproperties=font_lg)
	plt.savefig("ontology/figures/rdoc_seed_sim.png", 
				dpi=250, bbox_inches="tight")
	plt.close()


def load_framework_circuit(new, dtm_bin, act_bin, framework, n_iter=10000):

	from statsmodels.stats.multitest import multipletests

	file = "ontology/circuits/circuits_{}.csv".format(framework)
	if not os.path.isfile(file):

		pmids = dom_scores.index.intersection(act_bin.index)
		dom_scores = score_lists(new, dtm_bin, label_var="DOMAIN")
		dom_links = compute_cooccurrences(act_bin.loc[pmids], dom_scores.loc[pmids])
		dom_links_null = compute_cooccurrences_null(act_bin, dom_scores, 
													n_iter=n_iter, verbose=True)

		p = pd.DataFrame(index=act_bin.columns, columns=dom_scores.columns)
		for i, struct in enumerate(act_bin.columns):
			for j, dom in enumerate(dom_scores.columns):
				obs = dom_links.values[i,j]
				null = dom_links_null[i,j,:]
				p.loc[struct, dom] = np.sum(null > obs) / float(n_iter_fw)

		fdr = multipletests(p.values.ravel(), method="fdr_bh")[1]
		fdr = pd.DataFrame(fdr.reshape(p.shape), 
						   index=act_bin.columns, columns=dom_scores.columns)

		dom_links_thres = dom_links[fdr < 0.01]
		dom_links_thres = dom_links_thres.fillna(0.0)
		dom_links_thres.to_csv(file)
	
	else:
		dom_links_thres = pd.read_csv(file, index_col=0, header=0)

	return dom_links_thres


