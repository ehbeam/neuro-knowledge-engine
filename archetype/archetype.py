#!/usr/bin/python3

import collections, os
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
import utilities


def compute_gen_obs(stats, framework, domains, dom2docs, archetypes, docs, pmids, 
					metric="dice", clf="_lr", path=""):

	from scipy.spatial.distance import cdist

	print("\t  Computing observed values")

	df_obs = pd.DataFrame(index=domains, columns=pmids)
	for dom in domains:
		dom_pmids = dom2docs[dom]
		dom_vecs = docs.loc[dom_pmids].values
		dom_arche = archetypes[dom].values.reshape(1, archetypes.shape[0])
		dom_sims = 1.0 - cdist(dom_vecs, dom_arche, metric=metric)
		df_obs.loc[dom, dom_pmids] = dom_sims[:,0]
	df_obs.to_csv("{}data/arche_obs_{}{}.csv".format(path, framework, clf))
	stats["obs"][framework] = df_obs

	return stats


def compute_gen_mean(stats, framework, domains, clf="_lr", path=""):

	df_mean = pd.DataFrame({"OBSERVED": stats["obs"][framework].mean(axis=1)}, index=domains)
	df_mean.to_csv("{}data/arche_mean_{}{}.csv".format(path, framework, clf))
	stats["mean"][framework] = df_mean

	return stats


def compute_gen_boot(stats, framework, domains, dom2docs, archetypes, docs, 
					 metric="dice", n_iter=1000, clf="_lr", path=""):

	from scipy.spatial.distance import cdist
	np.random.seed(42)

	print("\t  Computing bootstrap distribution")

	file_boot = "{}data/arche_boot_{}{}_{}iter.csv".format(path, framework, clf, n_iter)
	
	if not os.path.isfile(file_boot):
		print("\tComputing bootstrap distribution")
		df_boot = np.zeros((len(domains), n_iter))
		for n in range(n_iter):
			boot = np.random.choice(range(len(docs.columns)), size=len(docs.columns), replace=True)
			for i, dom in enumerate(domains):
				dom_pmids = dom2docs[dom]
				dom_vecs = docs.loc[dom_pmids].values[:,boot]
				dom_arche = archetypes.values[boot,i].reshape(1, archetypes.shape[0])
				df_boot[i,n] = 1.0 - np.mean(cdist(dom_vecs, dom_arche, metric=metric))
			if n % int(n_iter / 10.0) == 0:
				print("\t\tProcessed {} iterations".format(n))
		df_boot = pd.DataFrame(df_boot, index=domains, columns=range(n_iter))
		df_boot.to_csv(file_boot)
	
	else:
		df_boot = pd.read_csv(file_boot, index_col=0, header=0)

	stats["boot"][framework] = df_boot

	return stats


def compute_gen_null(stats, framework, domains, dom2docs, archetypes, docs, 
					 metric="dice", n_iter=1000, clf="_lr", path=""):

	from scipy.spatial.distance import cdist
	np.random.seed(42)

	print("\t  Computing null distribution")

	file_null = "{}data/arche_null_{}{}_{}iter.csv".format(path, framework, clf, n_iter)
	
	if not os.path.isfile(file_null):
		print("\tComputing null distribution")
		df_null = np.zeros((len(domains), n_iter))
		for n in range(n_iter):
			null = np.random.choice(range(len(docs.columns)), 
									size=len(docs.columns), replace=False)
			for i, dom in enumerate(domains):
				dom_pmids = dom2docs[dom]
				dom_vecs = docs.loc[dom_pmids].values
				dom_arche = archetypes.values[null,i].reshape(1, archetypes.shape[0])
				df_null[i,n] = 1.0 - np.mean(cdist(dom_vecs, dom_arche, metric=metric))
			if n % int(n_iter / 10.0) == 0:
				print("\t\tProcessed {} iterations".format(n))
		df_null = pd.DataFrame(df_null, index=domains, columns=range(n_iter))
		df_null.to_csv(file_null)
	
	else:
		df_null = pd.read_csv(file_null, index_col=0, header=0)

	stats["null"][framework] = df_null

	return stats


def compute_gen_stats(stats, framework, lists, dom2docs, archetypes, docs, pmids, 
					  metric="dice", n_iter=1000, alpha=0.001, clf="_lr", path=""):

	domains = list(collections.OrderedDict.fromkeys(lists["DOMAIN"]))
	
	stats = compute_gen_obs(stats, framework, domains, dom2docs, archetypes, docs, pmids, metric=metric, clf=clf, path=path)
	stats = compute_gen_mean(stats, framework, domains, clf=clf, path=path)
	stats = compute_gen_boot(stats, framework, domains, dom2docs, archetypes, docs, metric=metric, n_iter=n_iter, clf=clf, path=path)
	stats = compute_gen_null(stats, framework, domains, dom2docs, archetypes, docs, metric=metric, n_iter=n_iter, clf=clf, path=path)
	stats["null_comparison"][framework] = utilities.compare_to_null(stats["null"][framework], stats["mean"][framework], n_iter, alpha=alpha)
	
	return stats

