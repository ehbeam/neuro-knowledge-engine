#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)

import sys
sys.path.append("..")
import utilities

def plot_violins(framework, domains, df, df_null, df_obs, palette, metric="mod",
				 dx=[], dy=0.5, ds=0.115, interval=0.999, alphas=[0.01, 0.001, 0.0001],
				 ylim=[0.5,8.5], yticks=[2,4,6,8], font=utilities.arial, print_fig=True, path=""):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams

	font_prop = font_manager.FontProperties(fname=font, size=20)
	rcParams["axes.linewidth"] = 1.5

	# Set up figure
	fig = plt.figure(figsize=(4.5, 2.1))
	ax = fig.add_axes([0,0,1,1])

	# Violin plot of observed values
	for i, dom in enumerate(domains):
		data = sorted(df_obs.loc[dom].dropna())
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
						 fontproperties=font_prop)

	# Confidence interval of null distribution
	n_iter = df_null.shape[1]
	lower = [sorted(df_null.loc[dom])[int(n_iter*(1.0-interval))] for dom in domains]
	upper = [sorted(df_null.loc[dom])[int(n_iter*interval)] for dom in domains]
	plt.fill_between(range(len(domains)), lower, upper, 
					 alpha=0.18, color="gray")
	plt.plot(df_null.values.mean(axis=1), linestyle="dashed", color="gray", linewidth=2)

	# Set plot parameters
	plt.xticks(range(len(domains)), [""]*len(domains))
	plt.yticks(yticks, fontproperties=font_prop)
	plt.xlim([-0.75, len(domains)-0.35])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	# Export figure
	plt.savefig("{}figures/{}_{}_{}iter.png".format(
				path, metric, framework, n_iter), dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()


def plot_framework_comparison(boot, obs, n_iter=1000, print_fig=True,
							  dx=0.38, ylim=[0.4,0.65], yticks=[], font=utilities.arial):
	
	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams
	
	font_lg = font_manager.FontProperties(fname=font, size=20)
	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=(2.1, 2.1))
	ax = fig.add_axes([0,0,1,1])

	i = 0
	labels = []
	for fw, dist in boot.items():
		labels.append(utilities.names[fw])
		dist_avg = np.mean(dist, axis=0)
		macro_avg = np.mean(obs[fw]["OBSERVED"])
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
	ax.set_xticklabels([], rotation=60, ha="right")
	plt.xticks(fontproperties=font_lg)
	plt.yticks(yticks, fontproperties=font_lg)
	plt.xlim([-0.75, len(boot.keys())-0.25])
	plt.ylim(ylim)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	plt.savefig("figures/mod_{}iter.png".format(n_iter), 
				dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()