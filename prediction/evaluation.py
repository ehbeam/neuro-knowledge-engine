#!/usr/bin/python3

import math, os, pickle
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
import utilities
from style import style
from prediction.neural_network import prediction


def load_fit(clf, framework, direction, opt_epochs=500, train_epochs=1000, path=""):

	if clf == "lr":
		fit = pickle.load(open("{}logistic_regression/fits/{}_{}.p".format(path, framework, direction), "rb"))
	
	if clf == "nn":

		from torch import load

		hyperparams = pd.read_csv("{}neural_network/data/params_{}_{}_{}epochs.csv".format(path, framework, direction, opt_epochs), 
								  header=None, index_col=0)
		h = {str(label): float(value) for label, value in hyperparams.iterrows()}
		state_dict = load("{}neural_network/fits/{}_{}_{}epochs.pt".format(path, framework, direction, train_epochs))
		layers = list(state_dict.keys())
		n_input = state_dict[layers[0]].shape[1]
		n_output = state_dict[layers[-2]].shape[0]
		fit = prediction.Net(n_input=n_input, n_output=n_output, 
							 n_hid=int(h["n_hid"]), p_dropout=h["p_dropout"])
		fit.load_state_dict(state_dict)

	return fit


def load_dataset(clf, scores, act, split, seed=42):

	if clf == "lr":
		X, Y = scores.loc[split].values, act.loc[split].values
	
	if clf == "nn":
		dataset = prediction.load_mini_batches(scores, act, split, mini_batch_size=len(split), seed=seed)[0]
		dataset = prediction.numpy2torch(dataset)
		X, Y = dataset
	
	return X, Y


def load_predictions(clf, fit, features):

	if clf == "lr":
		pred_probs = fit.predict_proba(features)

	if clf == "nn":

		from torch import no_grad
		
		with no_grad():
			pred_probs = fit.eval()(features).numpy()
	
	preds = 1 * pred_probs > 0.5
	
	return pred_probs, preds


def plot_curves(metric, framework, direction, x, y, palette, 
				diag=True, opacity=0.5, font=style.font, path="", print_fig=True):

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


def compute_eval_obs(stats, framework, direction, labels, pred_probs, preds):

	from sklearn.metrics import roc_auc_score, f1_score

	stats["obs"][framework][direction] = {}
	stats["obs"][framework][direction]["rocauc"] = compute_eval_metric(labels, pred_probs, roc_auc_score)
	stats["obs"][framework][direction]["f1"] = compute_eval_metric(labels, preds, f1_score)

	return stats


def compute_eval_boot(stats, framework, direction, labels, pred_probs, preds, ids, n_iter=1000, path=""):

	from sklearn.metrics import roc_auc_score, f1_score

	np.random.seed(42)

	stats["boot"][framework][direction] = {}
	stats["boot"][framework][direction]["rocauc"] = np.empty((len(stats["obs"][framework][direction]["rocauc"]), n_iter))
	stats["boot"][framework][direction]["f1"] = np.empty((len(stats["obs"][framework][direction]["f1"]), n_iter))
	
	rocauc_file = "{}data/rocauc_boot_{}_{}_{}iter.csv".format(path, framework, direction, n_iter)
	if os.path.isfile(rocauc_file):
		stats["boot"][framework][direction]["rocauc"] = pd.read_csv(rocauc_file, index_col=0, header=0).values
	else:
		for n in range(n_iter):
			samp = np.random.choice(range(len(ids)), size=len(ids), replace=True)
			stats["boot"][framework][direction]["rocauc"][:,n] = compute_eval_metric(labels[samp,:], pred_probs[samp,:], roc_auc_score)
	
	f1_file = "{}data/f1_boot_{}_{}_{}iter.csv".format(path, framework, direction, n_iter)
	if os.path.isfile(f1_file):
		stats["boot"][framework][direction]["f1"] = pd.read_csv(f1_file, index_col=0, header=0).values
	else:
		for n in range(n_iter):
			samp = np.random.choice(range(len(ids)), size=len(ids), replace=True)
			stats["boot"][framework][direction]["f1"][:,n] = compute_eval_metric(labels[samp,:], preds[samp,:], f1_score)

	return stats


def compute_eval_null(stats, framework, direction, labels, pred_probs, preds, ids, n_iter=1000, path=""):

	from sklearn.metrics import roc_auc_score, f1_score

	np.random.seed(42)

	stats["null"][framework][direction] = {}
	stats["null"][framework][direction]["rocauc"] = np.empty((len(stats["obs"][framework][direction]["rocauc"]), n_iter))
	stats["null"][framework][direction]["f1"] = np.empty((len(stats["obs"][framework][direction]["f1"]), n_iter))
	
	rocauc_file = "{}data/rocauc_null_{}_{}_{}iter.csv".format(path, framework, direction, n_iter)
	if os.path.isfile(rocauc_file):
		stats["null"][framework][direction]["rocauc"] = pd.read_csv(rocauc_file, index_col=0, header=0).values
	else:
		for n in range(n_iter):
			shuf = np.random.choice(range(len(ids)), size=len(ids), replace=False)
			stats["null"][framework][direction]["rocauc"][:,n] = compute_eval_metric(labels[shuf,:], pred_probs, roc_auc_score)
			if n % (n_iter/10) == 0:
				print("\t  Processed {} iterations".format(n))
	
	f1_file = "{}data/f1_null_{}_{}_{}iter.csv".format(path, framework, direction, n_iter)
	if os.path.isfile(f1_file):
		stats["null"][framework][direction]["f1"] = pd.read_csv(f1_file, index_col=0, header=0).values
	else:
		for n in range(n_iter):
			samp = np.random.choice(range(len(ids)), size=len(ids), replace=True)
			stats["null"][framework][direction]["f1"][:,n] = compute_eval_metric(labels[shuf,:], preds, f1_score)

	return stats


def compute_eval_null_ci(stats, framework, direction, n_iter=1000, interval=0.999, metric_labels=["rocauc", "f1"]):

	idx_lower = int((1.0-interval)*n_iter)
	idx_upper = int(interval*n_iter)
	stats["null_ci"][direction] = {}

	for metric in metric_labels:
		dist = stats["null"][framework][direction][metric]
		n_clf = dist.shape[0]
		stats["null_ci"][direction][metric] = {}
		stats["null_ci"][direction][metric]["lower"] = [sorted(dist[i,:])[idx_lower] for i in range(n_clf)]
		stats["null_ci"][direction][metric]["upper"] = [sorted(dist[i,:])[idx_upper] for i in range(n_clf)]
		stats["null_ci"][direction][metric]["mean"] = [np.mean(dist[i,:]) for i in range(n_clf)]

	return stats


def compute_eval_null_comparison(stats, framework, direction, n_iter=1000, interval=0.999, metric_labels=["rocauc", "f1"]):

	from statsmodels.stats.multitest import multipletests

	stats["p"][direction] = {}
	for metric in metric_labels:
		dist = stats["null"][framework][direction][metric]
		n_clf = dist.shape[0]
		stats["p"][direction][metric] = [np.sum(dist[i,:] >= stats["obs"][framework][direction][metric][i]) / float(n_iter) for i in range(n_clf)]

	stats["fdr"][direction] = {}
	for metric in metric_labels:
		stats["fdr"][direction][metric] = multipletests(stats["p"][direction][metric], method="fdr_bh")[1]

	return stats


def export_eval_stats(stats, framework, direction, metric_labels, index, n_iter=1000, path=""):

	for metric in metric_labels:
		for dist, dic in zip(["boot", "null"], [stats["boot"], stats["null"]]):
			df = pd.DataFrame(dic[framework][direction][metric], index=index[direction], columns=range(n_iter))
			df.to_csv("{}data/{}_{}_{}_{}_{}iter.csv".format(path, metric, dist, framework, direction, n_iter))
		obs_df = pd.DataFrame(stats["obs"][framework][direction][metric], index=index[direction])
		obs_df.to_csv("{}data/{}_obs_{}_{}.csv".format(path, metric, framework, direction), header=None)


def compute_eval_stats(stats, framework, direction, features, labels, pred_probs, preds, ids, index,
					   n_iter=1000, interval=0.999, metric_labels=["rocauc", "f1"], path="logistic_regression/"):
	
	stats = compute_eval_obs(stats, framework, direction, labels, pred_probs, preds)
	stats = compute_eval_boot(stats, framework, direction, labels, pred_probs, preds, ids, n_iter=n_iter, path=path)
	stats = compute_eval_null(stats, framework, direction, labels, pred_probs, preds, ids, n_iter=n_iter, path=path)
	stats = compute_eval_null_ci(stats, framework, direction, n_iter=n_iter, interval=interval, metric_labels=metric_labels)
	stats = compute_eval_null_comparison(stats, framework, direction, n_iter=n_iter, interval=interval, metric_labels=metric_labels)
	
	export_eval_stats(stats, framework, direction, metric_labels, index, n_iter=n_iter, path=path)

	return stats


def plot_eval_metric(metric, framework, direction, obs, boot, null_ci, fdr, 
					 palette, labels=[], dx=0.375, dxs=0.1, figsize=(11,4.5), 
					 ylim=[0.43,0.78], yticks=np.array(range(0,100,5))/100,
					 alphas=[0.05,0.01,0.001], font=style.font, print_fig=True, path=""):

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
		for alpha, y in zip(alphas, [0, 0.015, 0.03]):
			if fdr[i] < alpha:
				plt.text(i-dxs, max(boot[i,:])+y, "*", fontproperties=font_lg)

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
			  diag=True, alpha=0.5, color="gray", path="", print_fig=True):

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
	if print_fig:
		plt.show()
	plt.close()


def compute_roc(labels, pred_probs):

	from sklearn.metrics import roc_curve

	fpr, tpr = [], []
	for i in range(labels.shape[1]):
		fpr_i, tpr_i, _ = roc_curve(labels[:,i], pred_probs[:,i], pos_label=1)
		fpr.append(fpr_i)
		tpr.append(tpr_i)
	
	return fpr, tpr


def compute_prc(labels, pred_probs):

	from sklearn.metrics import precision_recall_curve

	precision, recall = [], []
	for i in range(labels.shape[1]):
		p_i, r_i, _ = precision_recall_curve(labels[:,i], pred_probs[:,i], pos_label=1)
		precision.append(p_i)
		recall.append(r_i)
	
	return precision, recall


def compare_frameworks(stats, frameworks, directions, metric_labels, n_iter=1000):

	from statsmodels.stats.multitest import multipletests

	fdr = {}
	for metric in metric_labels:
		fdr[metric] = {}
		for direction in directions:
			p = np.empty((len(frameworks), len(frameworks)))
			for i, fw_i in enumerate(frameworks):
				for j, fw_j in enumerate(frameworks):
					boot_i = np.mean(stats["boot"][fw_i][direction][metric], axis=0)
					boot_j = np.mean(stats["boot"][fw_j][direction][metric], axis=0)
					p[i,j] = np.sum((boot_i - boot_j) <= 0.0) / float(n_iter)
			fdr_md = multipletests(p.ravel(), method="fdr_bh")[1].reshape(p.shape)
			fdr[metric][direction] = pd.DataFrame(fdr_md, index=frameworks, columns=frameworks)
	
	return fdr


def plot_framework_comparison(metric, direction, boot, n_iter=1000, font=style.font,
							  dx=0.38, ylim=[0.4,0.65], yticks=[], path="", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_prop = font_manager.FontProperties(fname=font, size=22)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=(2.6,3.2))
	ax = fig.add_axes([0,0,1,1])

	i = 0
	labels = ["Data-Driven", "RDoC", "DSM"]
	for fw, dist in boot.items():
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


def map_framework_comparison(stats, metric_labels, n_iter=1000, alpha=0.001):

	from statsmodels.stats.multitest import multipletests

	dif_thres = {}
	for framework in ["rdoc", "dsm"]:
		dif_thres[framework] = {}
		for metric in metric_labels:
			dif_null = stats["null"]["data-driven"]["forward"][metric] - stats["null"][framework]["forward"][metric]
			dif_obs = np.array(stats["obs"]["data-driven"]["forward"][metric]) - np.array(stats["obs"][framework]["forward"][metric])
			pvals = [np.sum(np.abs(dif_null[i,:]) > np.abs(dif_obs[i])) / n_iter for i in range(len(dif_obs))]
			fdrs = multipletests(pvals, method="fdr_bh")[1]
			dif_obs[fdrs >= alpha] = np.nan
			dif_thres[framework][metric] = pd.DataFrame(dif_obs)
			dif_thres[framework][metric].columns = [""]
	
	return dif_thres

