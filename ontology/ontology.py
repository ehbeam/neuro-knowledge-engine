#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
from utilities import *

from matplotlib import font_manager, rcParams
font_md = font_manager.FontProperties(fname=arial, size=20)
font_lg = font_manager.FontProperties(fname=arial, size=22)
rcParams["axes.linewidth"] = 1.5

from scipy.spatial.distance import cdist


def load_ontology(k):
	list_file = "lists/lists_k{:02d}_oplen.csv".format(k)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "circuits/circuits_k{:02d}.csv".format(k)
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
	

def compute_eval_scores(scoring_function, directions, circuit_counts, features, fits, ids):
	eval_scores = {direction: np.zeros((len(circuit_counts))) for direction in directions}
	for i, k in enumerate(circuit_counts):

		y_pred_for = fits["forward"][k].predict(features[k]["function"].loc[ids])
		y_true_for = features[k]["structure"].loc[ids]
		y_pred_rev = fits["reverse"][k].predict(features[k]["structure"].loc[ids])
		y_true_rev = features[k]["function"].loc[ids]

		score_for = scoring_function(y_true_for.values, y_pred_for, average=None)
		eval_scores["forward"][i] = np.mean(score_for)
		score_rev = scoring_function(y_true_rev.values, y_pred_rev, average=None)
		eval_scores["reverse"][i] = np.mean(score_rev)

	eval_scores["mean"] = np.mean([eval_scores["forward"], eval_scores["reverse"]], axis=0)
	return eval_scores


def compute_eval_boot(scoring_function, directions, circuit_counts, 
					  features, fits, ids, n_iter=1000):
	eval_boot = {direction: np.zeros((len(circuit_counts), n_iter)) for direction in directions}
	for i, k in enumerate(circuit_counts):

		if i % 10 == 0:
			print("Processing {}th k".format(i))

		y_pred_for = fits["forward"][k].predict(features[k]["function"].loc[ids])
		y_true_for = features[k]["structure"].loc[ids]
		y_pred_rev = fits["reverse"][k].predict(features[k]["structure"].loc[ids])
		y_true_rev = features[k]["function"].loc[ids]

		for n in range(n_iter):
			boot = np.random.choice(range(len(ids)), size=len(ids), replace=True)
			score_for = scoring_function(y_true_for.values[boot,:], y_pred_for[boot,:], average=None)
			eval_boot["forward"][i,n] = np.mean(score_for)
			score_rev = scoring_function(y_true_rev.values[boot,:], y_pred_rev[boot,:], average=None)
			eval_boot["reverse"][i,n] = np.mean(score_rev)

	eval_boot["mean"] = np.mean([eval_boot["forward"], eval_boot["reverse"]], axis=0)
	return eval_boot


def compute_eval_null(scoring_function, directions, circuit_counts,
					  features, fits, ids, n_iter=1000):
	eval_null = {direction: np.zeros((len(circuit_counts), n_iter)) for direction in directions}
	for i, k in enumerate(circuit_counts):

		if i % 10 == 0:
			print("Processing {}th k".format(i))

		y_pred_for = fits["forward"][k].predict(features[k]["function"].loc[ids])
		y_true_for = features[k]["structure"].loc[ids]
		y_pred_rev = fits["reverse"][k].predict(features[k]["structure"].loc[ids])
		y_true_rev = features[k]["function"].loc[ids]

		for n in range(n_iter):
			null = np.random.choice(range(len(ids)), size=len(ids), replace=False)
			score_for = scoring_function(y_true_for.values[null,:], y_pred_for, average=None)
			eval_null["forward"][i,n] = np.mean(score_for)
			score_rev = scoring_function(y_true_rev.values[null,:], y_pred_rev, average=None)
			eval_null["reverse"][i,n] = np.mean(score_rev)

	eval_null["mean"] = np.mean([eval_null["forward"], eval_null["reverse"]], axis=0)
	return eval_null


def load_confidence_interval(distribution, direction, circuit_counts, interval=0.999):
	n_iter = float(distribution[direction].shape[1])
	lower = [sorted(distribution[direction][k,:])[int(n_iter*(1.0-interval))] for k in range(len(circuit_counts))]
	upper = [sorted(distribution[direction][k,:])[int(n_iter*interval)] for k in range(len(circuit_counts))]
	return lower, upper


def plot_scores(scores, direction, circuit_counts, boot, null, label, 
				shape="o", ylim=[0,1], op_k=7, interval=0.999):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_axes([0,0,1,1])

	# Plot distribution
	null_lower, null_upper = load_confidence_interval(null, direction, circuit_counts, interval=interval)
	plt.fill_between(circuit_counts, null_lower, null_upper, 
					 alpha=0.2, color="gray")
	plt.plot(circuit_counts, np.mean(null[direction], axis=1), 
			 linestyle="dashed", c="k", alpha=1, linewidth=2)

	# Plot bootstrap distribution
	n_iter = float(boot[direction].shape[1])
	boot_lower, boot_upper = load_confidence_interval(boot, direction, circuit_counts, interval=interval)
	for j, k in enumerate(circuit_counts):
		plt.plot([k, k], [boot_lower[j], boot_upper[j]], c="k", 
				 linewidth=9, alpha=0.15)

	# Plot observed values
	plt.scatter(circuit_counts, scores[direction], color="black", 
				marker=shape, zorder=1000, s=45, alpha=0.3)
	xp = np.linspace(circuit_counts[0], circuit_counts[-1], 100)
	idx = np.isfinite(scores[direction])
	p = np.poly1d(np.polyfit(np.array(circuit_counts)[idx], scores[direction][idx], 2))
	plt.plot(xp, p(xp), zorder=0, linewidth=2, 
			 color="black", alpha=1)

	# Plot selected value
	op_idx = list(circuit_counts).index(op_k)
	plt.scatter(circuit_counts[op_idx]+0.03, scores[direction][op_idx]-0.00015, 
				linewidth=2.5, edgecolor="black", color="none", 
				marker=shape, s=70, zorder=100)

	plt.xlim([0,41])
	plt.ylim(ylim)
	plt.xticks(fontproperties=font_md)
	plt.yticks(fontproperties=font_md)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)

	n_iter = boot[direction].shape[1]
	plt.savefig("figures/data-driven_{}_{}_{}iter.png".format(label, direction, n_iter), 
				dpi=250, bbox_inches="tight")
	plt.show()
	

def nounify(word, dtm):
	import word_forms
	from nltk.corpus import wordnet
	words = wordnet.synsets(word)
	if len(words) > 0:
		if words[0].pos() != "n":
			nouns = list(word_forms.get_word_forms(word)["n"])
			freqs = [dtm[noun].sum() if noun in dtm.columns else 0 for noun in nouns]
			if len(nouns) > 0:
				return str(nouns[freqs.index(max(freqs))])
	return word


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
		

def plot_wordclouds(framework, domains, lists, dtm):
	
	from wordcloud import WordCloud
	import matplotlib.pyplot as plt

	for i, dom in enumerate(domains):
		print(dom)
		
		def color_func(word, font_size, position, orientation, 
					   random_state=None, idx=0, **kwargs):
			return palettes[framework][i]

		tkns = lists.loc[lists["DOMAIN"] == dom, "TOKEN"]
		freq = dtm[tkns].sum().values
		tkns = [t.replace("_", " ") for t in tkns]

		cloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGB", 
						  max_font_size=100, prefer_horizontal=1, scale=20, margin=3,
						  width=550, height=15*len(tkns)+550, font_path=arial, 
						  random_state=42).generate_from_frequencies(zip(tkns, freq))

		fig = plt.figure(1, figsize=(2,10))
		plt.axis("off")
		plt.imshow(cloud.recolor(color_func=color_func, random_state=42))
		file_name = "figures/lists/{}_wordcloud_{}.png".format(framework, dom)
		plt.savefig(file_name, 
					dpi=800, bbox_inches="tight")
		transparent_background(file_name)
		plt.show()
