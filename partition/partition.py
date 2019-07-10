#!/usr/bin/python3

import os
import pandas as pd
import numpy as np


def load_archetypes(lists, circuits):

	import collections

	domains = list(collections.OrderedDict.fromkeys(lists["DOMAIN"]))
	words = sorted(list(set(lists["TOKEN"])))
	structures = sorted(list(set(circuits.index)))
	archetypes = pd.DataFrame(0.0, index=words+structures, columns=domains)
	for dom in domains:
		for word in lists.loc[lists["DOMAIN"] == dom, "TOKEN"]:
			archetypes.loc[word, dom] = 1.0
		for struct in structures:
			archetypes.loc[struct, dom] = circuits.loc[struct, dom]
	archetypes[archetypes > 0.0] = 1.0
	
	return archetypes


def load_partition(framework, clf, archetypes, docs):

	from scipy.spatial.distance import cdist

	partition_file = "partition/data/doc2dom_{}{}.csv".format(framework, clf)
	if not os.path.isfile(partition_file):

		dom_dists = cdist(docs.values, archetypes.values.T, metric="dice")

		pmids = docs.index
		doc2dom = {pmid: 0 for pmid in pmids}
		for i, pmid in enumerate(pmids):
			doc2dom[pmid] = np.argmin(dom_dists[i,:]) + 1

		partition_df = pd.Series(doc2dom)
		partition_df.to_csv(partition_file, header=False)

	else:
		partition_df = pd.read_csv(partition_file, header=None, index_col=0)
		doc2dom = {int(pmid): int(dom) for pmid, dom in partition_df.iterrows()}

	domains = list(archetypes.columns)
	dom2docs = {dom: [] for dom in domains}
	for doc, dom in doc2dom.items():
		dom2docs[domains[dom-1]].append(doc)

	return doc2dom, dom2docs


def compute_distances(docs, metric="dice"):

	from scipy.spatial.distance import cdist

	dists = cdist(docs, docs, metric=metric)
	dists = pd.DataFrame(dists, index=docs.index, columns=docs.index)	
	
	return dists


def plot_partition(framework, doc_dists, transitions, path="", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams

	rcParams["axes.linewidth"] = 1

	fig = plt.figure(figsize=(10,10), frameon=False)
	ax = fig.add_axes([0,0,1,1])

	X = doc_dists.values.astype(np.float)
	im = ax.matshow(X, cmap=cm.Greys_r, vmin=0, vmax=1, alpha=1) 
	plt.xticks(transitions)
	plt.yticks(transitions)
	ax.set_xticklabels([])
	ax.set_yticklabels([])

	plt.savefig("{}figures/partition_{}.png".format(path, framework), 
				dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()

