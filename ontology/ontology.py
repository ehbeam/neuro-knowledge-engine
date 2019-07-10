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
import utilities
from style import style
from prediction.neural_network.sherlock.neural_network import Net

from matplotlib import font_manager, rcParams
font_md = font_manager.FontProperties(fname=style.font, size=20)
font_lg = font_manager.FontProperties(fname=style.font, size=22)
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


def assign_functions(k, clust, splits, act, dtm, lexicon, list_lens=range(5,26)):

	from scipy.stats import pointbiserialr

	lists = pd.DataFrame()
	for i in range(k):
		structures = list(clust.loc[clust["CLUSTER"] == i+1, "STRUCTURE"])
		centroid = np.mean(act.loc[splits["train"], structures], axis=1)
		R = pd.Series([pointbiserialr(dtm.loc[splits["train"], word], centroid)[0] 
					   for word in lexicon], index=lexicon)
		R = R[R > 0].sort_values(ascending=False)[:max(list_lens)]
		R = pd.DataFrame({"CLUSTER": [i+1 for l in range(max(list_lens))], 
						  "TOKEN": R.index, "R": R.values})
		lists = lists.append(R)
	return lists


def load_fits(clf, directions, n_circuits, path=""):

	import pickle, torch

	fits = {}
	for direction in directions:
		fits[direction] = {}
		for k in n_circuits:
			if clf == "lr":
				fit_file = "{}logistic_regression/sherlock/fits/{}_k{:02d}_{}.p".format(path, direction, k, direction)
				fits[direction][k] = pickle.load(open(fit_file, "rb"))
			if clf == "nn":
				state_dict = torch.load("{}neural_network/sherlock/fits/{}_k{:02d}.pt".format(path, direction, k))
				hyperparams = pd.read_csv("{}neural_network/data/params_data-driven_k{:02d}_{}.csv".format(path, k, direction), 
										  header=None, index_col=0)
				h = {str(label): float(value) for label, value in hyperparams.iterrows()}
				layers = list(state_dict.keys())
				n_input = state_dict[layers[0]].shape[1]
				n_output = state_dict[layers[-2]].shape[0]
				fits[direction][k] = Net(n_input=n_input, n_output=n_output, 
								 		 n_hid=int(h["n_hid"]), p_dropout=h["p_dropout"])
				fits[direction][k].load_state_dict(state_dict)
	return fits


def load_domain_features(dtm, act, directions, n_circuits, suffix="", path=""):

	from sklearn.preprocessing import binarize

	features = {k: {} for k in n_circuits}
	for k in n_circuits:
		domains = range(1, k+1)
		lists, circuits = load_ontology(k, suffix=suffix, path=path)
		function_features = pd.DataFrame(index=dtm.index, columns=domains)
		structure_features = pd.DataFrame(index=act.index, columns=domains)
		for i in domains:
			functions = lists.loc[lists["CLUSTER"] == i, "TOKEN"]
			function_features[i] = dtm[functions].sum(axis=1)
			structures = circuits.loc[circuits["CLUSTER"] == i, "STRUCTURE"]
			structure_features[i] = act[structures].sum(axis=1)
		function_features = pd.DataFrame(utilities.doc_mean_thres(function_features), 
										 index=dtm.index, columns=domains)
		structure_features = pd.DataFrame(binarize(structure_features), 
										 index=act.index, columns=domains)
		features[k]["function"] = function_features
		features[k]["structure"] = structure_features
	return features


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


def compute_eval_scores(clf, scoring_function, directions, n_circuits, features, fits, ids):

	eval_scores = {direction: np.zeros((len(n_circuits))) for direction in directions}
	
	for i, k in enumerate(n_circuits):
		
		if clf == "lr":
			function_features = features[k]["function"].loc[ids] 
			structure_features = features[k]["structure"].loc[ids]
			y_pred_for = fits["forward"][k].predict_proba(function_features)
			y_pred_rev = fits["reverse"][k].predict_proba(structure_features)

		if clf == "nn":
			with torch.no_grad():
				data_set = load_mini_batches(features[k]["function"].loc[ids], features[k]["structure"].loc[ids], 
											 ids, mini_batch_size=len(ids), seed=42)
				function_features, structure_features = numpy2torch(data_set[0])
				y_pred_for = fits["forward"][k].eval()(function_features).float()
				y_pred_rev = fits["reverse"][k].eval()(structure_features).float()

		eval_scores["forward"][i] = scoring_function(structure_features, y_pred_for, average="macro")
		eval_scores["reverse"][i] = scoring_function(function_features, y_pred_rev, average="macro")

	eval_scores["mean"] = np.mean([eval_scores["forward"], eval_scores["reverse"]], axis=0)
	return eval_scores


def load_eval_data(features, k, ids):
	with torch.no_grad():
		data_set = load_mini_batches(features[k]["function"].loc[ids], features[k]["structure"].loc[ids], 
									ids, mini_batch_size=len(ids), seed=42)
		function_features, structure_features = numpy2torch(data_set[0])
	return function_features, structure_features


def load_eval_preds(clf, fit, features):
	
	if clf == "lr":
		preds = fit.predict_proba(features)

	if clf == "nn":
		with torch.no_grad():
			preds = fit.eval()(features).float()

	return preds


def compute_eval_boot(clf, scoring_function, directions, n_circuits, 
					  features, fits, ids, n_iter=1000, verbose=True, path=""):
	eval_boot = {direction: np.zeros((len(n_circuits), n_iter)) for direction in directions}

	file_for = "{}data/circuits_{}_forward_boot_{}iter.csv".format(path, clf, n_iter)
	if not os.path.exists(file_for):
		print("Bootstrap for N Domains | Forward")
		for i, k in enumerate(n_circuits):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			function_features, structure_features = load_eval_data(features, k, ids)
			y_pred_for = load_eval_preds(clf, fits["forward"][k], function_features)
			y_true_for = structure_features
			for n in range(n_iter):
				boot = np.random.choice(range(len(ids)), size=len(ids), replace=True)
				score_for = scoring_function(y_true_for[boot,:], y_pred_for[boot,:], average="macro")
				eval_boot["forward"][i,n] = score_for
		pd.DataFrame(eval_boot["forward"]).to_csv(file_for)
	elif os.path.exists(file_for):
		eval_boot["forward"] = pd.read_csv(file_for, index_col=0, header=0).values

	file_rev = "{}data/circuits_{}_reverse_boot_{}iter.csv".format(path, clf, n_iter)
	if not os.path.exists(file_rev):
		print("Bootstrap for N Domains | Reverse")
		for i, k in enumerate(n_circuits):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			with torch.no_grad():
				function_features, structure_features = load_eval_data(features, k, ids)
				y_pred_rev = load_eval_preds(clf, fits["reverse"][k], structure_features)
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


def compute_eval_null(clf, scoring_function, directions, n_circuits,
					  features, fits, ids, n_iter=1000, verbose=True, path=""):
	eval_null = {direction: np.zeros((len(n_circuits), n_iter)) for direction in directions}
	
	file_for = "{}data/circuits_{}_forward_null_{}iter.csv".format(path, clf, n_iter)
	if not os.path.exists(file_for):
		print("Permutation for N Domains | Forward")
		for i, k in enumerate(n_circuits):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			with torch.no_grad():
				function_features, structure_features = load_eval_data(features, k, ids)
				y_pred_for = load_eval_preds(clf, fits["forward"][k], function_features)
				y_true_for = structure_features
			for n in range(n_iter):
				null = np.random.choice(range(len(ids)), size=len(ids), replace=False)
				score_for = scoring_function(y_true_for[null,:], y_pred_for, average="macro")
				eval_null["forward"][i,n] = score_for
		pd.DataFrame(eval_null["forward"]).to_csv(file_for)
	elif os.path.exists(file_for):
		eval_null["forward"] = pd.read_csv(file_for, index_col=0, header=0).values

	file_rev = "{}data/circuits_{}_reverse_null_{}iter.csv".format(path, clf, n_iter)
	if not os.path.exists(file_rev):
		print("Permutation for N Domains | Reverse")
		for i, k in enumerate(n_circuits):
			if i % 10 == 0 and verbose:
				print("   Processing {}th k".format(i))
			with torch.no_grad():
				function_features, structure_features = load_eval_data(features, k, ids)
				y_pred_rev = load_eval_preds(clf, fits["reverse"][k], structure_features)
				y_true_rev = function_features
			for n in range(n_iter):
				null = np.random.choice(range(len(ids)), size=len(ids), replace=False)
				score_rev = scoring_function(y_true_rev[null,:], y_pred_rev, average="macro")
				eval_null["reverse"][i,n] = score_rev
		pd.DataFrame(eval_null["reverse"]).to_csv(file_rev)
		print("")
	elif os.path.exists(file_rev):
		eval_null["reverse"] = pd.read_csv(file_rev, index_col=0, header=0).values

	eval_null["mean"] = np.mean([eval_null["forward"], eval_null["reverse"]], axis=0)
	return eval_null


def plot_scores(direction, n_circuits, stats, shape="o", op_k=6, interval=0.999, 
				ylim=[0.45,0.7], yticks=np.arange(0.4,0.75,0.05), 
				font=style.font, clf="lr", path="", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_axes([0,0,1,1])
	font_prop = font_manager.FontProperties(fname=font, size=20)

	# Plot null distribution
	null_lower, null_upper = load_confidence_interval(stats["null"], direction, n_circuits, interval=interval)
	plt.fill_between(n_circuits, null_lower, null_upper, 
					 alpha=0.2, color="gray")
	plt.plot(n_circuits, np.mean(stats["null"][direction][:len(n_circuits)], axis=1), 
			 linestyle="dashed", c="k", alpha=1, linewidth=2)

	# Plot bootstrap distribution
	n_iter = float(stats["boot"][direction].shape[1])
	boot_lower, boot_upper = load_confidence_interval(stats["boot"], direction, n_circuits, interval=interval)
	for j, k in enumerate(n_circuits):
		plt.plot([k, k], [boot_lower[j], boot_upper[j]], c="k", 
				 linewidth=9, alpha=0.15)

	# Plot observed values
	plt.scatter(n_circuits, stats["scores"][direction][:len(n_circuits)], color="black", 
				marker=shape, zorder=1000, s=45, alpha=0.3)
	xp = np.linspace(n_circuits[0], n_circuits[-1], 100)
	idx = np.isfinite(stats["scores"][direction][:len(n_circuits)])
	p = np.poly1d(np.polyfit(np.array(n_circuits)[idx], stats["scores"][direction][:len(n_circuits)][idx], 2))
	plt.plot(xp, p(xp), zorder=0, linewidth=2, 
			 color="black", alpha=1)

	# Plot selected value
	op_idx = list(n_circuits).index(op_k)
	plt.scatter(n_circuits[op_idx]+0.03, stats["scores"][direction][:len(n_circuits)][op_idx]-0.00015, 
				linewidth=2.5, edgecolor="black", color="none", 
				marker=shape, s=70, zorder=100)

	plt.xlim([0,max(n_circuits)+1])
	plt.ylim(ylim)
	plt.xticks(fontproperties=font_prop)
	plt.yticks(yticks,fontproperties=font_prop)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)

	n_iter = stats["boot"][direction].shape[1]
	plt.savefig("{}figures/data-driven_{}_rocauc_{}_{}iter.png".format(path, clf, direction, n_iter), 
				dpi=250, bbox_inches="tight")
	if print_fig:
	   plt.show()
	plt.close()


def load_confidence_interval(distribution, direction, n_circuits, interval=0.999):
	n_iter = float(distribution[direction].shape[1])
	lower = [sorted(distribution[direction][k,:])[int(n_iter*(1.0-interval))] for k in range(len(n_circuits))]
	upper = [sorted(distribution[direction][k,:])[int(n_iter*interval)] for k in range(len(n_circuits))]
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


def export_ontology(lists, circuits, n_domains, ord_domains, clf, act, k2name, path=""):

	names = [k2name[k] for k in ord_domains]
	k2order = {k: ord_domains.index(k)+1 for k in range(1, n_domains+1)}

	lists["ORDER"] = [k2order[k] for k in lists["CLUSTER"]]
	lists["DOMAIN"] = [k2name[k] for k in lists["CLUSTER"]]
	lists = lists.sort_values(["ORDER", "R"], ascending=[True, False])
	lists = lists[["ORDER", "CLUSTER", "DOMAIN", "TOKEN", "R", "ROC_AUC"]]
	lists.to_csv("{}lists/lists_data-driven_{}.csv".format(path, clf), index=None)

	circuits["ORDER"] = [k2order[k] for k in circuits["CLUSTER"]]
	circuits["DOMAIN"] = [k2name[k] for k in circuits["CLUSTER"]]
	circuits = circuits.sort_values(["ORDER", "STRUCTURE"])
	circuits = circuits[["ORDER", "CLUSTER", "DOMAIN", "STRUCTURE"]]
	circuits.to_csv("{}circuits/clusters_data-driven_{}.csv".format(path, clf), index=None)

	circuit_mat = pd.DataFrame(0.0, index=act.columns, columns=names)
	for name in names:
		structures = circuits.loc[circuits["DOMAIN"] == name, "STRUCTURE"]
		for structure in structures:
			circuit_mat.loc[structure, name] = 1.0
	circuit_mat.to_csv("{}circuits/circuits_data-driven_{}.csv".format(path, clf))

	return lists, circuit_mat


def plot_wordclouds(framework, domains, lists, dtm, path="", 
					font=style.font, print_fig=True, width=550):
	
	from wordcloud import WordCloud
	import matplotlib.pyplot as plt

	for i, dom in enumerate(domains):
		
		def color_func(word, font_size, position, orientation, 
					   random_state=None, idx=0, **kwargs):
			return style.palettes[framework][i]

		tkns = lists.loc[lists["DOMAIN"] == dom, "TOKEN"]
		freq = dtm[tkns].sum().values
		tkns = [t.replace("_", " ") for t in tkns]

		cloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGB", 
						  max_font_size=100, prefer_horizontal=1, scale=20, margin=3,
						  width=width, height=15*len(tkns)+550, font_path=font, 
						  random_state=42).generate_from_frequencies(zip(tkns, freq))

		fig = plt.figure()
		plt.axis("off")
		plt.imshow(cloud.recolor(color_func=color_func, random_state=42))
		file_name = "{}figures/lists/{}_wordcloud_{}.png".format(path, framework, dom)
		plt.savefig(file_name, 
					dpi=800, bbox_inches="tight")
		utilities.transparent_background(file_name)
		if print_fig:
			print(dom)
			plt.show()
		plt.close()


def load_rdoc_lists(lexicon, vsm, seeds, dtm, n_terms=range(5,26), path="", verbose=False):
	
	from collections import OrderedDict
	from scipy.spatial.distance import cdist

	n_thres = max(list(n_terms))

	lists = pd.DataFrame()
	doms = list(OrderedDict.fromkeys(seeds["DOMAIN"]))
	for l, dom in enumerate(doms):
		dom_df = seeds.loc[seeds["DOMAIN"] == dom]
		tokens = list(dom_df["TOKEN"])
		centroid = np.mean(vsm.loc[tokens]).values.reshape(1, -1)
		dists = cdist(vsm.loc[lexicon], centroid, metric="cosine")
		dists = pd.Series(dists.reshape(-1), index=lexicon).sort_values()
		dists = [(w, d) for w, d in dists.iteritems()][:n_thres]
		
		if verbose:
			if len(dists) == 0:
				print("No tokens assigned to {}".format(dom))
			if len(dists) < n_thres:
				print("{} tokens assigned to {}".format(len(dists), dom))
		
		for w, d in dists:
			dic = {"ORDER": [l],
				   "DOMAIN": [dom],
				   "TOKEN": [w],
				   "SOURCE": ["RDoC" if w in tokens else "Lexicon"],
				   "DISTANCE": [d]}
			lists = lists.append(pd.DataFrame(dic))
	
	lists = lists[["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]]
	lists = lists.sort_values(["ORDER", "DISTANCE"])
	lists = lists.loc[lists["TOKEN"].isin(dtm.columns)]
	lists.to_csv("{}lists/lists_rdoc.csv".format(path), index=None)
	op_df = load_optimized_lists(doms, lists, n_terms, seeds, vsm)
	op_df.to_csv("{}data/df_rdoc_opsim.csv".format(path))
	lists = update_lists(doms, op_df, lists, "rdoc", path=path)

	return lists


def load_dsm_lists(lexicon, vsm, seeds, n_terms=range(5, 26), path="", verbose=False):
	
	from collections import OrderedDict
	from scipy.spatial.distance import cdist

	n_thres = max(list(n_terms))

	doms = list(OrderedDict.fromkeys(seeds["DOMAIN"]))
	tokens = []
	for dom in doms:
		tokens += list(set(seeds.loc[seeds["DOMAIN"] == dom, "TOKEN"]))
	unique = [tkn for tkn in tokens if tokens.count(tkn) == 1]

	lists = pd.DataFrame()
	for dom in doms:
		dom_df = seeds.loc[seeds["DOMAIN"] == dom]
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
				print("No tokens assigned to {}".format(dom))
		
		for w, d in dists:
			dic = {"ORDER": [list(dom_df["ORDER"])[0] + 1],
				   "DOMAIN": [dom],
				   "TOKEN": [w],
				   "SOURCE": ["DSM" if w in tokens else "Lexicon"],
				   "DISTANCE": [d]}
			lists = lists.append(pd.DataFrame(dic))
	
	lists = lists[["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]]
	lists = lists.sort_values(["ORDER", "DISTANCE"])
	lists.to_csv("{}lists/lists_dsm.csv".format(path), index=None)

	op_df = load_optimized_lists(doms, lists, n_terms, seeds, vsm)
	op_df.to_csv("{}data/df_dsm_opsim.csv".format(path))
	lists = update_lists(doms, op_df, lists, "dsm", path=path)

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


def update_lists(doms, op_df, lists, framework, path=""):
	columns = ["ORDER", "DOMAIN", "TOKEN", "SOURCE", "DISTANCE"]
	new = pd.DataFrame(columns=columns)
	for order, dom in enumerate(doms):
		list_len = op_df.loc[dom, "OPTIMAL"]
		dom_df = lists.loc[lists["DOMAIN"] == dom][:list_len]
		new = new.append(dom_df)
	new.to_csv("{}lists/lists_{}_opsim.csv".format(path, framework), index=None)
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
					plt.text(i-0.1325, max(data) + y, "*", fontproperties=font_xlg)


def compute_rdoc_similarity(doms, seeds_rdoc, lists_rdoc, vsm_rdoc, n_iter=1000, interval=0.999, path=""):

	from statsmodels.stats.multitest import multipletests
	np.random.seed(42)
	
	mccoy = pd.read_csv("{}lists/lists_cqh.csv".format(path), index_col=None)
	
	seeds_centroid = compute_centroid(seeds_rdoc, doms, vsm_rdoc)
	lists_centroid = compute_centroid(lists_rdoc, doms, vsm_rdoc)
	mccoy_centroid = compute_centroid(mccoy, doms, vsm_rdoc)

	stats = {"lists": {}, "mccoy": {}}

	stats["lists"]["boot"], stats["mccoy"]["boot"] = np.empty((len(doms), n_iter)), np.empty((len(doms)-1, n_iter))
	for n in range(n_iter):
		stats["lists"]["boot"][:,n] = 1.0 - compute_sims_sample(lists_centroid, seeds_centroid, vsm_rdoc)
		stats["mccoy"]["boot"][:(len(doms)-1),n] = (1.0 - compute_sims_sample(mccoy_centroid, seeds_centroid, vsm_rdoc))[:(len(doms)-1)]

	stats["lists"]["null"] = np.empty((len(doms), n_iter))
	for n in range(n_iter):
		stats["lists"]["null"][:,n] = 1.0 - compute_sims_shuffle(lists_centroid, seeds_centroid, vsm_rdoc)
		
	stats["lists"]["obs"] = np.reshape(1.0 - compute_sims(lists_rdoc, seeds_centroid, doms, vsm_rdoc), (len(doms), 1))
	stats["mccoy"]["obs"] = np.reshape(1.0 - compute_sims(mccoy, seeds_centroid, doms, vsm_rdoc), (len(doms), 1))

	pvals_dif = np.sum(np.less(stats["lists"]["boot"][:(len(doms)-1),:] - stats["mccoy"]["boot"], 0.0), axis=1) / n_iter
	stats["fdrs"] = multipletests(pvals_dif, method="fdr_bh")[1] # Benjamini-Hochberg

	stats["lists"]["lower"] = [sorted(stats["lists"]["null"][i,:])[int(n_iter*(1.0-interval))] for i in range(len(doms))]
	stats["lists"]["upper"] = [sorted(stats["lists"]["null"][i,:])[int(n_iter*interval)] for i in range(len(doms))]
	
	return stats


def plot_rdoc_similarity(doms, stats, alphas=[0.01, 0.001], font=style.font, path="ontology/"):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams
	
	font_lg = font_manager.FontProperties(fname=font, size=20)
	font_xlg = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=(2.9, 4.5))
	ax = fig.add_axes([0,0,1,1])
	plt.plot(range(len(doms)), stats["lists"]["null"].mean(axis=1),
			 "gray", linestyle="dashed", linewidth=2)
	plt.fill_between(range(len(doms)), stats["lists"]["lower"], y2=stats["lists"]["upper"], 
					 color="gray", alpha=0.2)
	plot_violin(ax, stats["mccoy"]["boot"], stats["mccoy"]["obs"], doms[:-1], ["k"]*5, alphas, font_xlg), 
	plot_violin(ax, stats["lists"]["boot"], stats["lists"]["obs"], doms, style.palettes["rdoc"], 
				alphas, font_xlg, fdrs=stats["fdrs"], sig=True)
	plt.xlim([-0.75, len(doms)-0.5])
	plt.ylim([-0.4, 1.1])
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	ax.set_xticks(range(len(doms)))
	ax.set_xticklabels([])
	plt.yticks(fontproperties=font_lg)
	plt.savefig("{}figures/rdoc_seed_sim.png".format(path), 
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


