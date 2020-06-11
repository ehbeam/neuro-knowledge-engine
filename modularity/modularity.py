#!/usr/bin/python3

import collections, os
import pandas as pd
import numpy as np

import sys
sys.path.append("..")
sys.path.append("../..")
import utilities


def sort_dists(domains, dom2docs, doc2dom, dists):

	sorted_pmids = []
	for dom in range(len(domains)):
		sorted_pmids += [pmid for pmid, sys in doc2dom.items() if sys == dom + 1]
	sorted_dists = dists[sorted_pmids].loc[sorted_pmids]

	dom_idx = {dom: {"min": 0, "max": 0} for dom in domains}
	for dom in domains:
		dom_pmids = dom2docs[dom]
		dom_idx[dom]["min"] = sorted_pmids.index(dom_pmids[0])
		dom_idx[dom]["max"] = sorted_pmids.index(dom_pmids[-1]) + 1

	return sorted_dists, dom_idx


def compute_mod_obs(stats, framework, domains, dom2docs, doc2dom, sorted_dists, dom_idx, pmids, clf="_lr", path=""):

	file_obs = "{}data/mod_obs_{}{}.csv".format(path, framework, clf)
	file_mean = "{}data/mod_mean_{}{}.csv".format(path, framework, clf)
	print("\t  Computing observed values")

	dists_int, dists_ext = {}, {}
	df_obs = pd.DataFrame(index=domains, columns=pmids)
	df_mean = pd.DataFrame(index=domains, columns=["OBSERVED"])

	for dom in domains:
		
		dom_min, dom_max = dom_idx[dom]["min"], dom_idx[dom]["max"]
		dom_dists = sorted_dists.values[:,dom_min:dom_max][dom_min:dom_max,:]
		dists_int[dom] = dom_dists.ravel()
		dist_int = np.nanmean(dom_dists)
		sorted_dists_int = np.mean(dom_dists, axis=0)
		
		other_dists_lower = sorted_dists.values[:,dom_min:dom_max][:dom_min,:]
		other_dists_upper = sorted_dists.values[:,dom_min:dom_max][dom_max:,:]
		other_dists = np.concatenate((other_dists_lower, other_dists_upper))
		dists_ext[dom] = other_dists.ravel()
		dist_ext = np.nanmean(other_dists)
		sorted_dists_ext = np.mean(other_dists, axis=0)
		
		df_mean.loc[dom, "OBSERVED"] = dist_ext / dist_int
		df_obs.loc[dom, dom2docs[dom]] = sorted_dists_ext / sorted_dists_int

	df_obs.to_csv(file_obs)
	df_mean.to_csv(file_mean)	

	stats["obs"][framework] = df_obs
	stats["mean"][framework] = df_mean

	return stats, dists_int, dists_ext


def compute_mod_boot(stats, framework, domains, dom2docs, dists_int, dists_ext, n_iter=1000, clf="_lr", path=""):

	np.random.seed(42)

	file_boot = "{}data/mod_boot_{}{}_{}iter.csv".format(path, framework, clf, n_iter)
	if not os.path.isfile(file_boot):
		print("\t  Computing bootstrap distribution")
		df_boot = np.empty((len(domains), n_iter))
		
		for n in range(n_iter):
			for i, dom in enumerate(domains):
				
				boot_int = np.random.choice(dists_int[dom], size=len(dists_int[dom]), replace=True)
				dist_int = np.nanmean(boot_int)
				
				boot_ext = np.random.choice(dists_ext[dom], size=len(dists_ext[dom]), replace=True)
				dist_ext = np.nanmean(boot_ext)
				
				df_boot[i,n] = dist_ext / dist_int
				
			if n % int(n_iter / 10.0) == 0:
				print("\t\tProcessed {} iterations".format(n))
				
		df_boot = pd.DataFrame(df_boot, index=domains, columns=range(n_iter))
		df_boot.to_csv(file_boot)
	
	else:
		df_boot = pd.read_csv(file_boot, index_col=0, header=0)

	stats["boot"][framework] = df_boot

	return stats


def compute_mod_null(stats, framework, domains, dom2docs, sorted_dists, dom_idx, n_iter=1000, clf="_lr", path=""):

	np.random.seed(42)

	file_null = "{}data/mod_null_{}{}_{}iter.csv".format(path, framework, clf, n_iter)

	if not os.path.isfile(file_null):
		print("\t  Computing null distribution")
		null_dists = sorted_dists.values.copy()
		df_null = np.empty((len(domains), n_iter))
		
		for n in range(n_iter):
			np.random.shuffle(null_dists)
			
			for i, dom in enumerate(domains):
				
				dom_min, dom_max = dom_idx[dom]["min"], dom_idx[dom]["max"]
				dom_dists = null_dists[:,dom_min:dom_max][dom_min:dom_max,:]
				dist_int = np.nanmean(dom_dists)

				other_dists_lower = null_dists[:,dom_min:dom_max][:dom_min,:]
				other_dists_upper = null_dists[:,dom_min:dom_max][dom_max:,:]
				other_dists = np.concatenate((other_dists_lower, other_dists_upper))
				dist_ext = np.nanmean(other_dists)
				
				df_null[i,n] = dist_ext / dist_int
				
			if n % int(n_iter / 10.0) == 0:
				print("\t\tProcessed {} iterations".format(n))
		
		df_null = pd.DataFrame(df_null, index=domains, columns=range(n_iter))
		df_null.to_csv(file_null)
	
	else:
		df_null = pd.read_csv(file_null, index_col=0, header=0)

	stats["null"][framework] = df_null

	return stats


def compute_mod_stats(stats, framework, lists, dom2docs, doc2dom, dists, pmids, clf="_lr", n_iter=1000, alpha=0.001, path=""):

	domains = list(collections.OrderedDict.fromkeys(lists["DOMAIN"]))
	sorted_dists, dom_idx = sort_dists(domains, dom2docs, doc2dom, dists)

	stats, dists_int, dists_ext = compute_mod_obs(stats, framework, domains, dom2docs, doc2dom, sorted_dists, dom_idx, pmids, clf=clf, path=path)
	stats = compute_mod_boot(stats, framework, domains, dom2docs, dists_int, dists_ext, n_iter=n_iter, clf=clf, path=path)
	stats = compute_mod_null(stats, framework, domains, dom2docs, sorted_dists, dom_idx, n_iter=n_iter, clf=clf, path=path)
	stats["null_comparison"][framework] = utilities.compare_to_null(stats["null"][framework], stats["mean"][framework], n_iter, alpha=alpha)
	
	return stats


def load_mod_data(framework, version=190325, suffix="", clf="", path="", verbose=True):

	from collections import OrderedDict
	from scipy.spatial.distance import cdist

	act_bin = utilities.load_coordinates(path="{}../data".format(path))
	dtm_bin = utilities.load_doc_term_matrix(version=version, binarize=True, path="{}../data".format(path))
	lists, circuits = utilities.load_framework(framework, suffix=suffix, clf=clf, path="{}../ontology".format(path))

	words = sorted(list(set(lists["TOKEN"])))
	structures = sorted(list(set(act_bin.columns)))
	domains = list(OrderedDict.fromkeys(lists["DOMAIN"]))
	
	archetypes = pd.DataFrame(0.0, index=words+structures, columns=domains)
	for dom in domains:
		for word in lists.loc[lists["DOMAIN"] == dom, "TOKEN"]:
			archetypes.loc[word, dom] = 1.0
		for struct in structures:
			archetypes.loc[struct, dom] = circuits.loc[struct, dom]
	archetypes[archetypes > 0.0] = 1.0

	pmids = dtm_bin.index.intersection(act_bin.index)

	dtm_words = dtm_bin.loc[pmids, words]
	act_structs = act_bin.loc[pmids, structures]

	docs = dtm_words.copy()
	docs[structures] = act_structs.copy()
	doc_dists = cdist(docs, docs, metric="dice")
	doc_dists = pd.DataFrame(doc_dists, index=pmids, columns=pmids)

	doc2dom_df = pd.read_csv("{}../partition/data/doc2dom_{}{}.csv".format(path, framework, clf), 
                         header=None, index_col=0)
	doc2dom = {int(pmid): int(dom) for pmid, dom in doc2dom_df.iterrows()}

	dom2docs = {dom: [] for dom in domains}
	for doc, dom in doc2dom.items():
		dom2docs[domains[dom-1]].append(doc)

	sorted_pmids = []
	for dom in range(len(domains)):
		sorted_pmids += [pmid for pmid, sys in doc2dom.items() if sys == dom + 1]

	doc_dists = doc_dists[sorted_pmids].loc[sorted_pmids]

	dom_idx = {dom: {"min": 0, "max": 0} for dom in domains}
	for dom in domains:
		dom_pmids = dom2docs[dom]
		dom_idx[dom]["min"] = sorted_pmids.index(dom_pmids[0])
		dom_idx[dom]["max"] = sorted_pmids.index(dom_pmids[-1]) + 1

	return domains, pmids, dom_idx, doc_dists, dom2docs


def run_mod_obs(framework, domains, pmids, dom_idx, doc_dists, dom2docs, clf="", path="../"):

	dists_int, dists_ext = {}, {}
	df_obs = pd.DataFrame(index=domains, columns=pmids)
	df_stat = pd.DataFrame(index=domains, columns=["OBSERVED"])

	for dom in domains:

		dom_min, dom_max = dom_idx[dom]["min"], dom_idx[dom]["max"]
		dom_dists = doc_dists.values[:,dom_min:dom_max][dom_min:dom_max,:]
		dists_int[dom] = dom_dists.ravel()
		dist_int = np.nanmean(dom_dists)
		doc_dists_int = np.mean(dom_dists, axis=0)

		other_dists_lower = doc_dists.values[:,dom_min:dom_max][:dom_min,:]
		other_dists_upper = doc_dists.values[:,dom_min:dom_max][dom_max:,:]
		other_dists = np.concatenate((other_dists_lower, other_dists_upper))
		dists_ext[dom] = other_dists.ravel()
		dist_ext = np.nanmean(other_dists)
		doc_dists_ext = np.mean(other_dists, axis=0)

		df_stat.loc[dom, "OBSERVED"] = dist_ext / dist_int
		df_obs.loc[dom, dom2docs[dom]] = doc_dists_ext / doc_dists_int

	df_obs.to_csv("{}data/mod_obs_{}{}.csv".format(path, framework, clf))
	
	return df_obs, df_stat, dists_int, dists_ext


def run_mod_boot(framework, version=190325, suffix="", clf="", n_iter=1000, verbose=True, path="../"):

	domains, pmids, dom_idx, doc_dists, dom2docs = load_mod_data(framework, version=version, suffix=suffix, clf=clf, verbose=True, path=path)
	df_obs, df_stat, dists_int, dists_ext = run_mod_obs(framework, domains, pmids, dom_idx, doc_dists, dom2docs, clf=clf, path=path)

	file_boot = "{}data/mod_boot_{}{}_{}iter.csv".format(path, framework, clf, n_iter)
	if not os.path.isfile(file_boot):
		df_boot = np.empty((len(domains), n_iter))

		for n in range(n_iter):
			for i, dom in enumerate(domains):

				boot_int = np.random.choice(dists_int[dom], size=len(dists_int[dom]), replace=True)
				dist_int = np.nanmean(boot_int)

				boot_ext = np.random.choice(dists_ext[dom], size=len(dists_ext[dom]), replace=True)
				dist_ext = np.nanmean(boot_ext)

				df_boot[i,n] = dist_ext / dist_int

			if verbose and n % int(n_iter / 10.0) == 0:
				print("Processed {} iterations".format(n))

		df_boot = pd.DataFrame(df_boot, index=domains, columns=range(n_iter))
		df_boot.to_csv(file_boot)

	else:
		df_boot = pd.read_csv(file_boot, index_col=0, header=0)

	return df_boot


def run_mod_null(framework, version=190325, suffix="", clf="", n_iter=1000, verbose=True, path="../"):

	domains, pmids, dom_idx, doc_dists, dom2docs = load_mod_data(framework, version=version, suffix=suffix, clf=clf, verbose=True, path=path)
	df_obs, df_stat, dists_int, dists_ext = run_mod_obs(framework, domains, pmids, dom_idx, doc_dists, dom2docs, clf=clf, path=path)

	null_dists = doc_dists.values.copy()
	file_null = "data/mod_null_{}{}_{}iter.csv".format(framework, clf, n_iter)
	if not os.path.isfile(file_null):

		df_null = np.empty((len(domains), n_iter))

		for n in range(n_iter):
			np.random.shuffle(null_dists)

			for i, dom in enumerate(domains):

				dom_min, dom_max = dom_idx[dom]["min"], dom_idx[dom]["max"]
				dom_dists = null_dists[:,dom_min:dom_max][dom_min:dom_max,:]
				dist_int = np.nanmean(dom_dists)

				other_dists_lower = null_dists[:,dom_min:dom_max][:dom_min,:]
				other_dists_upper = null_dists[:,dom_min:dom_max][dom_max:,:]
				other_dists = np.concatenate((other_dists_lower, other_dists_upper))
				dist_ext = np.nanmean(other_dists)

				df_null[i,n] = dist_ext / dist_int

			if verbose and n % int(n_iter / 10.0) == 0:
				print("Processed {} iterations".format(n))

		df_null = pd.DataFrame(df_null, index=domains, columns=range(n_iter))
		df_null.to_csv(file_null)

	else:
		df_null = pd.read_csv(file_null, index_col=0, header=0)

	return df_null

