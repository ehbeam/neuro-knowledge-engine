#!/usr/bin/python3

import os
import pandas as pd
import numpy as np


def load_docs(dtm, act, words):

	docs = dtm[words].copy()
	docs[list(act.columns)] = act.copy()

	return docs


def load_archetypes(lists, circuits, domains, words):

	structures = sorted(list(set(circuits.index)))
	archetypes = pd.DataFrame(0.0, index=words+structures, columns=domains)
	for dom in domains:
		for word in lists.loc[lists["DOMAIN"] == dom, "TOKEN"]:
			archetypes.loc[word, dom] = 1.0
		for struct in structures:
			archetypes.loc[struct, dom] = circuits.loc[struct, dom]
	archetypes[archetypes > 0.0] = 1.0
	
	return archetypes


def load_partition(framework, clf, splits, archetypes, docs, path=""):

	from scipy.spatial.distance import cdist

	partition_file = "{}partition/data/doc2dom_{}{}.csv".format(path, framework, clf)
	if not os.path.isfile(partition_file):

		dom_dists = cdist(docs.values, archetypes.values.T, metric="dice")
		dom_dists = pd.DataFrame(dom_dists, index=docs.index, columns=domains)

		pmids = docs.index
		doc2dom = {pmid: 0 for pmid in pmids}
		for i, pmid in enumerate(pmids):
			doc2dom[pmid] = dom_dists.columns[np.argmin(dom_dists.values[i,:])]
		
		partition_df = pd.Series(doc2dom)
		partition_df.to_csv(partition_file, header=False)

	else:
		doc2dom_df = pd.read_csv(partition_file, header=None, index_col=0)
		doc2dom = {int(pmid): str(dom.values[0]) for pmid, dom in doc2dom_df.iterrows()}

	domains = list(archetypes.columns)
	dom2docs = {dom: {split: [] for split in ["discovery", "replication"]} for dom in domains}
	for doc, dom in doc2dom.items():
		for split, split_pmids in splits.items():
			if doc in split_pmids:
				dom2docs[dom][split].append(doc)

	return doc2dom, dom2docs


def compute_distances(docs, metric="dice"):

	from scipy.spatial.distance import cdist

	dists = cdist(docs, docs, metric=metric)
	dists = pd.DataFrame(dists, index=docs.index, columns=docs.index)
	
	return dists


def plot_partition(framework, doc_dists, transitions, palette, figsize=(4,4), 
				   linewidth=1.5, path="", suffix="", print_fig=True):

	import matplotlib.pyplot as plt
	from matplotlib import cm, font_manager, rcParams
	import matplotlib.patches as patches

	rcParams["axes.linewidth"] = 1.5

	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0,0,1,1])

	X = doc_dists.values.astype(np.float)
	im = ax.imshow(X, cmap=cm.Greys_r, vmin=0, vmax=1, alpha=1)
		
	plt.xticks(transitions)
	plt.yticks(transitions)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.xaxis.set_tick_params(width=1.5, length=7, top=False)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	
	for i, t in enumerate(transitions[:-1]):
		t_next = transitions[i+1]
		lines = [[t,t], [t,t_next], [t_next,t_next]]
		ax.add_patch(patches.Polygon(lines, alpha=0.4, linewidth=3,
									 facecolor=palette[i], edgecolor=palette[i]))

	plt.savefig("{}figures/partition_{}{}.png".format(path, framework, suffix), 
				dpi=250, bbox_inches="tight")
	if print_fig:
		plt.show()
	plt.close()

