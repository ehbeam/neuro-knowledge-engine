#!/usr/bin/python3

import sys
sys.path.append("..")
import utilities


def run_mds(metric=True, eps=0.001, max_iter=300):

	import pandas as pd
	from scipy.spatial.distance import cdist
	from sklearn.manifold import MDS

	act_bin = utilities.load_coordinates()

	frameworks = ["data-driven", "rdoc", "dsm"]
	suffixes = ["", "_opsim", "_opsim"]
	lists, words = {}, []
	for framework in frameworks:
		lists[framework] = pd.read_csv("../ontology/lists/lists_{}{}.csv".format(
									   framework, utilities.suffix[framework]), index_col=None)
		words += list(lists[framework]["TOKEN"])

	words = sorted(list(set(words)))
	dtm_bin = utilities.load_doc_term_matrix(version=190325, binarize=True)

	pmids = act_bin.index.intersection(dtm_bin.index)

	act_bin = act_bin.loc[pmids]
	dtm_bin = dtm_bin.loc[pmids]

	vecs = act_bin.copy()
	vecs[words] = dtm_bin[words]

	doc_dists = cdist(vecs, vecs, metric="dice")

	mds = MDS(n_components=2, max_iter=max_iter, metric=metric, eps=eps,
			  dissimilarity="precomputed", random_state=42)

	outfile = "data/mds_metric{}_eps{}_iter{}.csv".format(
			  int(metric), eps, int(max_iter))
	X = mds.fit_transform(doc_dists)
	X_df = pd.DataFrame(X, index=vecs.index, columns=["X", "Y"])
	X_df.to_csv(outfile)


def plot_mds(X, framework, colors, markers, metric=True, eps=0.001, max_iter=5000, 
			 path="", suffix="", print_fig=True):

	import matplotlib.pyplot as plt

	fig, ax = plt.subplots(figsize=(8,8))

	key = framework + suffix

	for i in range(X.shape[0]):
		plt.scatter(X[i,0], X[i,1], c=colors[key][i], 
					marker=markers[key][i], alpha=0.525, s=13)
	plt.xticks([])
	plt.yticks([])
	for side in ["top", "bottom", "left", "right"]:
		ax.spines[side].set_visible(False)
	plt.tight_layout()
	plt.savefig("{}figures/{}_mds_metric{}_eps{}_iter{}_vecs.png".format(
				path, key, int(metric), eps, int(max_iter)), dpi=250)
	if print_fig:
		plt.show()
	plt.close()


