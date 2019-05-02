#!/usr/bin/python3

import pandas as pd
import numpy as np
from collections import OrderedDict
from scipy.spatial.distance import cdist
from utilities import *

def compute_null(framework, version=190124, suffix="", n_iter=10000):

	# Load the data
	act_bin = load_coordinates()
	dtm_bin = load_doc_term_matrix(version=version, binarize=True)
	lists, circuits = load_framework(framework, suffix=suffix)
	words = sorted(list(set(lists["TOKEN"])))
	structures = sorted(list(set(act_bin.columns)))
	domains = list(OrderedDict.fromkeys(lists["DOMAIN"]))

	# Domain archetypes
	archetypes = pd.DataFrame(0.0, index=words+structures, columns=domains)
	for dom in domains:
		for word in lists.loc[lists["DOMAIN"] == dom, "TOKEN"]:
			archetypes.loc[word, dom] = 1.0
		for struct in structures:
			archetypes.loc[struct, dom] = circuits.loc[struct, dom]
	archetypes[archetypes > 0.0] = 1.0

	# Document structure-term vectors
	pmids = dtm_bin.index.intersection(act_bin.index)
	dtm_words = dtm_bin.loc[pmids, words]
	act_structs = act_bin.loc[pmids, structures]
	docs = dtm_words.copy()
	docs[structures] = act_structs.copy()

	# Document assignments
	doc2dom_df = pd.read_csv("../partition/data/doc2dom_{}.csv".format(framework), 
						 header=None, index_col=0)
	doc2dom = {int(pmid): int(dom) for pmid, dom in doc2dom_df.iterrows()}
	dom2docs = {dom: [] for dom in domains}
	for doc, dom in doc2dom.items():
		dom2docs[domains[dom-1]].append(doc)

	# Compute null distribution
	df_null = np.zeros((len(domains), n_iter))
	for n in range(n_iter):
		null = np.random.choice(range(len(docs.columns)), 
								size=len(docs.columns), replace=False)
		for i, dom in enumerate(domains):
			dom_pmids = dom2docs[dom]
			dom_vecs = docs.loc[dom_pmids].values
			dom_arche = archetypes.values[null,i].reshape(1, archetypes.shape[0])
			df_null[i,n] = 1.0 - np.mean(cdist(dom_vecs, dom_arche, metric="dice"))
		if n % int(n_iter / 10.0) == 0:
			print("Processed {} iterations".format(n))

	# Export results
	file_null = "data/arche_null_{}_{}iter.csv".format(framework, n_iter)
	df_null = pd.DataFrame(df_null, index=domains, columns=range(n_iter))
	df_null.to_csv(file_null)