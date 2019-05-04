#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
from utilities import *


def plot_violins(framework, domains, df, df_null, df_boot, palette, metric="mod",
				 dx=[], dy=0.5, ds=0.115, interval=0.999, alphas=[0.01, 0.001, 0.0001],
				 ylim=[0.5,8.5], yticks=[2,4,6,8]):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams

	font = font_manager.FontProperties(fname=arial, size=20)
	font_lg = font_manager.FontProperties(fname=arial, size=24)
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
						 fontproperties=font_lg)

	# Confidence interval of null distribution
	n_iter = df_null.shape[1]
	lower = [sorted(df_null.loc[dom])[int(n_iter*(1.0-interval))] for dom in domains]
	upper = [sorted(df_null.loc[dom])[int(n_iter*interval)] for dom in domains]
	plt.fill_between(range(len(domains)), lower, upper, 
					 alpha=0.18, color="gray")
	plt.plot(df_null.mean(axis=1), linestyle="dashed", color="gray", linewidth=2)

	# Set plot parameters
	plt.xticks(range(len(domains)), [""]*len(domains))
	plt.yticks(yticks, fontproperties=font)
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