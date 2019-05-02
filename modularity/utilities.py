#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)


c = {"red": "#CE7D69", "orange": "#BA7E39", "yellow": "#CEBE6D", 
	 "chartreuse": "#AEC87C", "green": "#77B58A", "blue": "#7597D0", 
	 "magenta": "#B07EB6", "purple": "#7D74A3", "brown": "#846B43"}

palettes = {"data-driven": [c["blue"], c["magenta"], c["yellow"], c["green"], c["red"], c["purple"], c["orange"]],
			"rdoc": [c["blue"], c["red"], c["green"], c["purple"], c["yellow"], c["orange"]],
			"dsm": [c["purple"], c["chartreuse"], c["orange"], c["blue"], c["red"], c["magenta"], c["yellow"], c["green"], c["brown"]]}


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_coordinates():
	atlas_labels = pd.read_csv("../data/brain/labels.csv")
	activations = pd.read_csv("../data/brain/coordinates.csv", index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]
	return activations


def load_doc_term_matrix(version=190124, binarize=True):
	dtm = pd.read_csv("../data/text/dtm_{}.csv.gz".format(version), compression="gzip", index_col=0)
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm


def load_framework(framework, suffix=""):
	list_file = "../ontology/lists/lists_{}{}.csv".format(framework, suffix)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "../ontology/circuits/circuits_{}.csv".format(framework)
	circuits = pd.read_csv(circuit_file, index_col=0)
	return lists, circuits


def plot_violins(framework, domains, df, df_null, df_boot, palette, metric="mod",
				 dx=[], dy=0.5, ds=0.115, interval=0.999, alphas=[0.01, 0.001, 0.0001],
				 ylim=[0.5,10.5], yticks=[2,4,6,8,10]):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams

	arial = "../style/Arial Unicode.ttf"
	prop = font_manager.FontProperties(fname=arial, size=20)
	prop_lg = font_manager.FontProperties(fname=arial, size=24)
	rcParams["axes.linewidth"] = 1.5

	# Set up figure
	fig = plt.figure(figsize=(4.5, 2.1))
	ax = fig.add_axes([0,0,1,1])

	# Violin plot of observed values
	for i, dom in enumerate(domains):
		data = sorted(df_boot.loc[dom].dropna())
		obs = df.loc[dom, "OBSERVED"]
		v = ax.violinplot(data, positions=[i], 
						  showmeans=False, showmedians=False, widths=0.85)
		for pc in v["bodies"]:
			pc.set_facecolor(palette[i])
			pc.set_edgecolor(palette[i])
			pc.set_linewidth(1.25)
			pc.set_alpha(0.4)
		for line in ["cmaxes", "cmins", "cbars"]:
			v[line].set_edgecolor("none")
		plt.plot([i-dx[i], i+dx[i]], [obs, obs], 
					c=palette[i], alpha=1, lw=2)

		# Comparison test
		dys = dy * np.array([0, 1, 2])
		for alpha, y in zip(alphas, dys):
			if df["FDR"][i] < alpha:
				plt.text(i-ds, min(max(data), 9.5) + y, "*", 
						 fontproperties=prop_lg)

	# Confidence interval of null distribution
	n_iter = df_null.shape[1]
	lower = [sorted(df_null.loc[dom])[int(n_iter*(1.0-interval))] for dom in domains]
	upper = [sorted(df_null.loc[dom])[int(n_iter*interval)] for dom in domains]
	plt.fill_between(range(len(domains)), lower, upper, 
					 alpha=0.18, color="gray")
	plt.plot(df_null.mean(axis=1), linestyle="dashed", color="gray", linewidth=2)

	# Set plot parameters
	plt.xticks(range(len(domains)), [""]*len(domains))
	plt.yticks(yticks, fontproperties=prop)
	plt.xlim([-0.75, len(domains)-0.35])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	# Export figure
	plt.savefig("figures/{}_{}_{}iter.png".format(
				metric, framework, n_iter), dpi=250, bbox_inches="tight")
	plt.show()