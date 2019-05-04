#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, f1_score

import sys
sys.path.append("..")
from utilities import *

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
arial = "../style/Arial Unicode.ttf"
font_sm = font_manager.FontProperties(fname=arial, size=8)
font_md = font_manager.FontProperties(fname=arial, size=20)
font_lg = font_manager.FontProperties(fname=arial, size=22)
prop_xlg = font_manager.FontProperties(fname=arial, size=28)
rcParams["axes.linewidth"] = 1.5


def compute_roc(X, Y, clf):
	probas_pred = clf.predict_proba(X)
	fpr, tpr = [], []
	for i in range(Y.shape[1]):
		fpr_i, tpr_i, _ = roc_curve(Y.values[:,i], 
									probas_pred[:,i], pos_label=1)
		fpr.append(fpr_i)
		tpr.append(tpr_i)
	return fpr, tpr


def compute_prc(X, Y, clf):
	probas_pred = clf.predict_proba(X)
	precision, recall = [], []
	for i in range(Y.shape[1]):
		p_i, r_i, _ = precision_recall_curve(Y.values[:,i], 
											 probas_pred[:,i], pos_label=1)
		precision.append(p_i)
		recall.append(r_i)
	return precision, recall


def plot_curves(metric, framework, direction, x, y, palette, diag=True):

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the curves
	for i in range(len(x)):
		plt.plot(x[i], y[i], alpha=0.55, 
				 c=palette[i], linewidth=2)

	# Plot a diagonal line
	if diag:
		plt.plot([-1,2], [-1,2], linestyle="dashed", c="k", 
				 alpha=1, linewidth=2)

	plt.xlim([-0.05, 1])
	plt.ylim([-0.05, 1])
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	plt.xticks(fontproperties=font_lg)
	plt.yticks(fontproperties=font_lg)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	plt.tight_layout()
	plt.savefig("figures/{}_{}_{}.png".format(metric, framework, direction), 
				bbox_inches="tight", dpi=250)
	plt.show()


def compute_eval_metric(X, Y, clf, metric_function):
	pred = clf.predict(X)
	metric_scores = []
	for i in range(Y.shape[1]):
		metric_scores.append(metric_function(Y.values[:,i], pred[:,i], average="macro"))
	return metric_scores


def plot_eval_metric(metric, framework, direction, obs, boot, null_ci, fdr, 
					 palette, labels=[], dx=0.375, dxs=0.1, dy=0.012, figsize=(11,4.5), 
					 ylim=[0.43,0.78], alphas=[0.001, 0.0001, 0.00001]):

	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0,0,1,1])

	# Plot the null distributions
	n_clf = len(obs)
	plt.fill_between(range(n_clf), null_ci["lower"], null_ci["upper"], 
					 alpha=0.2, color="gray")
	plt.plot(null_ci["mean"], 
			 linestyle="dashed", color="gray", linewidth=2)

	# Plot the bootstrap distributions
	dys = dy * np.array([0, 1, 2])
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
		for alpha, y in zip(alphas, dys):
			if fdr[i] < alpha:
				plt.text(i-dxs, max(boot[i,:]) + y, "*", fontproperties=font_lg)

	ax.set_xticks(range(n_clf))
	ax.set_xticklabels(labels, rotation=60, ha="right")
	plt.xticks(fontproperties=font_sm)
	plt.yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75], fontproperties=font_lg)
	plt.xlim([-1, n_clf])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	n_iter = boot.shape[1]
	plt.savefig("figures/{}_{}_{}_{}iter.png".format(metric, framework, direction, n_iter), 
				dpi=250, bbox_inches="tight")