#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)
from scipy.spatial.distance import cdist
from matplotlib import font_manager, rcParams
from collections import OrderedDict

arial = "../style/Arial Unicode.ttf"
font_md = font_manager.FontProperties(fname=arial, size=20)
font_lg = font_manager.FontProperties(fname=arial, size=22)
rcParams["axes.linewidth"] = 1.5


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


def load_lexicon(sources):
	lexicon = []
	for source in sources:
		file = "../lexicon/lexicon_{}.txt".format(source)
		lexicon += [token.strip() for token in open(file, "r").readlines()]
	return sorted(lexicon)


def load_ontology(k):
	list_file = "lists/lists_k{:02d}_oplen.csv".format(k)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "circuits/circuits_k{:02d}.csv".format(k)
	circuits = pd.read_csv(circuit_file, index_col=None)
	return lists, circuits


def observed_over_expected(df):
	# From https://github.com/cgpotts/cs224u/blob/master/vsm.py
	col_totals = df.sum(axis=0)
	total = col_totals.sum()
	row_totals = df.sum(axis=1)
	expected = np.outer(row_totals, col_totals) / total
	oe = df / expected
	return oe


def pmi(df, positive=True):
	# From https://github.com/cgpotts/cs224u/blob/master/vsm.py
	df = observed_over_expected(df)
	with np.errstate(divide="ignore"):
		df = np.log(df)
	df[np.isinf(df)] = 0.0  # log(0) = 0
	if positive:
		df[df < 0] = 0.0
	return df


def compute_cooccurrences(activations, scores):
	X = np.matmul(activations.values.T, scores.values)
	X = pmi(X, positive=True)
	X = pd.DataFrame(X, columns=scores.columns, index=activations.columns)
	X = X.dropna(axis=1, how="any")
	X = X.loc[:, (X != 0).any(axis=0)]
	return X


def compute_cooccurrences_null(activations, scores, n_iter=1000, verbose=False):
	np.random.seed(42)
	X_null = np.empty((activations.shape[1], scores.shape[1], n_iter))
	act_mat = activations.values.T
	scores_mat = scores.values
	n_docs = len(scores)
	for n in range(n_iter):
		null = np.random.choice(range(n_docs), size=n_docs, replace=False)
		X = np.matmul(act_mat, scores_mat[null,:])
		X = pmi(X, positive=True)
		X_null[:,:,n] = X
		if verbose:
			if n % (n_iter/10) == 0:
				print("Iteration {}".format(n))
	return X_null
	

def compute_eval_scores(scoring_function, directions, features, fits, ids, 
						circuit_counts=range(2,41)):
	eval_scores = {direction: np.zeros((len(circuit_counts))) for direction in directions}
	for i, k in enumerate(circuit_counts):

		y_pred_for = fits["forward"][k].predict(features[k]["function"].loc[ids])
		y_true_for = features[k]["structure"].loc[ids]
		y_pred_rev = fits["reverse"][k].predict(features[k]["structure"].loc[ids])
		y_true_rev = features[k]["function"].loc[ids]

		score_for = scoring_function(y_true_for.values, y_pred_for, average=None)
		eval_scores["forward"][i] = np.mean(score_for)
		score_rev = scoring_function(y_true_rev.values, y_pred_rev, average=None)
		eval_scores["reverse"][i] = np.mean(score_rev)

	eval_scores["mean"] = np.mean([eval_scores["forward"], eval_scores["reverse"]], axis=0)
	return eval_scores


def compute_eval_boot(scoring_function, directions, features, fits, ids,
					  circuit_counts=range(2,41), n_iter=1000):
	eval_boot = {direction: np.zeros((len(circuit_counts), n_iter)) for direction in directions}
	for i, k in enumerate(circuit_counts):

		if i % 10 == 0:
			print("Processing {}th k".format(i))

		y_pred_for = fits["forward"][k].predict(features[k]["function"].loc[ids])
		y_true_for = features[k]["structure"].loc[ids]
		y_pred_rev = fits["reverse"][k].predict(features[k]["structure"].loc[ids])
		y_true_rev = features[k]["function"].loc[ids]

		for n in range(n_iter):
			boot = np.random.choice(range(len(ids)), size=len(ids), replace=True)
			score_for = scoring_function(y_true_for.values[boot,:], y_pred_for[boot,:], average=None)
			eval_boot["forward"][i,n] = np.mean(score_for)
			score_rev = scoring_function(y_true_rev.values[boot,:], y_pred_rev[boot,:], average=None)
			eval_boot["reverse"][i,n] = np.mean(score_rev)

	eval_boot["mean"] = np.mean([eval_boot["forward"], eval_boot["reverse"]], axis=0)
	return eval_boot


def compute_eval_null(scoring_function, directions, features, fits, ids, 
					  circuit_counts=range(2,41), n_iter=1000):
	eval_null = {direction: np.zeros((len(circuit_counts), n_iter)) for direction in directions}
	for i, k in enumerate(circuit_counts):

		if i % 10 == 0:
			print("Processing {}th k".format(i))

		y_pred_for = fits["forward"][k].predict(features[k]["function"].loc[ids])
		y_true_for = features[k]["structure"].loc[ids]
		y_pred_rev = fits["reverse"][k].predict(features[k]["structure"].loc[ids])
		y_true_rev = features[k]["function"].loc[ids]

		for n in range(n_iter):
			null = np.random.choice(range(len(ids)), size=len(ids), replace=False)
			score_for = scoring_function(y_true_for.values[null,:], y_pred_for, average=None)
			eval_null["forward"][i,n] = np.mean(score_for)
			score_rev = scoring_function(y_true_rev.values[null,:], y_pred_rev, average=None)
			eval_null["reverse"][i,n] = np.mean(score_rev)

	eval_null["mean"] = np.mean([eval_null["forward"], eval_null["reverse"]], axis=0)
	return eval_null


def load_confidence_interval(distribution, direction, interval=0.999, circuit_counts=range(2, 41)):
	n_iter = float(distribution[direction].shape[1])
	lower = [sorted(distribution[direction][k,:])[int(n_iter*(1.0-interval))] for k in range(len(circuit_counts))]
	upper = [sorted(distribution[direction][k,:])[int(n_iter*interval)] for k in range(len(circuit_counts))]
	return lower, upper


def plot_scores(scores, direction, boot, null, label, shape="o", 
				ylim=[0,1], circuit_counts=range(2, 41)):
	
	import matplotlib.pyplot as plt
	from matplotlib import font_manager, rcParams

	fig = plt.figure(figsize=(4, 3))
	ax = fig.add_axes([0,0,1,1])

	# Plot distribution
	null_lower, null_upper = load_confidence_interval(null, direction)
	plt.fill_between(circuit_counts, null_lower, null_upper, 
					 alpha=0.2, color="gray")
	plt.plot(circuit_counts, np.mean(null[direction], axis=1), 
			 linestyle="dashed", c="k", alpha=1, linewidth=2)

	# Plot bootstrap distribution
	n_iter = float(boot[direction].shape[1])
	boot_lower, boot_upper = load_confidence_interval(boot, direction)
	for j, k in enumerate(circuit_counts):
		plt.plot([k, k], [boot_lower[j], boot_upper[j]], c="k", 
				 linewidth=9, alpha=0.15)

	# Plot observed values
	plt.scatter(circuit_counts, scores[direction], color="black", 
				marker=shape, zorder=1000, s=45, alpha=0.3)
	xp = np.linspace(circuit_counts[0], circuit_counts[-1], 100)
	idx = np.isfinite(scores[direction])
	p = np.poly1d(np.polyfit(np.array(circuit_counts)[idx], scores[direction][idx], 2))
	plt.plot(xp, p(xp), zorder=0, linewidth=2, 
			 color="black", alpha=1)

	# Plot selected value
	op_idx = np.argmax(scores["mean"][:12])
	plt.scatter(circuit_counts[op_idx]+0.03, scores[direction][op_idx]-0.00015, 
				linewidth=2.5, edgecolor="black", color="none", 
				marker=shape, s=70, zorder=100)

	plt.xlim([0,41])
	plt.ylim(ylim)
	plt.xticks(fontproperties=font_md)
	plt.yticks(fontproperties=font_md)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	
	n_iter = boot[direction].shape[1]
	plt.savefig("figures/data-driven_{}_{}_{}iter.png".format(label, direction, n_iter), 
				dpi=250, bbox_inches="tight")
	plt.show()


def nounify(word, dtm):
	import word_forms
	from nltk.corpus import wordnet
	words = wordnet.synsets(word)
	if len(words) > 0:
		if words[0].pos() != "n":
			nouns = list(word_forms.get_word_forms(word)["n"])
			freqs = [dtm[noun].sum() if noun in dtm.columns else 0 for noun in nouns]
			if len(nouns) > 0:
				return str(nouns[freqs.index(max(freqs))])
	return word


def term_degree_centrality(i, lists, circuits, dtm, ids, reweight=False):
	terms = list(set(lists.loc[lists["CLUSTER"] == i, "TOKEN"]))
	ttm = pd.DataFrame(np.matmul(dtm.loc[ids, terms].T, dtm.loc[ids, terms]), 
					   index=terms, columns=terms)
	adj = pd.DataFrame(0, index=terms, columns=terms)
	for term_i in terms:
		for term_j in terms:
			adj.loc[term_i, term_j] = ttm.loc[term_i, term_j]
	degrees = adj.sum(axis=1)
	degrees = degrees.loc[terms]
	degrees = degrees.sort_values(ascending=False)
	return degrees


def compute_centroid(df, labels, vsm, level="DOMAIN"):
	centroids = []
	for i, label in enumerate(labels):
		tkns = df.loc[df[level] == label, "TOKEN"]
		tkns = [tkn for tkn in tkns if tkn in vsm.index]
		centroids.append(np.mean(vsm.loc[tkns]))
	return np.array(centroids)


def compute_sims_sample(df, seed_centroid, vsm):
	idx = np.random.choice(range(vsm.shape[1]), size=vsm.shape[1], replace=True)
	sims = cdist(seed_centroid[:,idx], df[:,idx], "cosine")
	return np.diagonal(sims)


def compute_sims_shuffle(df, seed_centroid, vsm):
	idx_i = np.random.choice(range(vsm.shape[1]), size=vsm.shape[1], replace=False)
	idx_j = np.random.choice(range(vsm.shape[1]), size=vsm.shape[1], replace=False)
	sims = cdist(seed_centroid[:,idx_i], df[:,idx_j], "cosine")
	return np.diagonal(sims)


def compute_sims(df, seed_centroid, labels, vsm, level="DOMAIN"):
	centroids = compute_centroid(df, labels, vsm, level=level)
	sims = cdist(seed_centroid, centroids, "cosine")
	return np.diagonal(sims) 


def report_significance(pvals, labels, alphas=[0.01, 0.001, 0.0001]):
    for p, lab in zip(pvals, labels):
        stars = "".join(["*" for alpha in alphas if p < alpha])
        print("{:22s} p={:6.6f} {}".format(lab, p, stars))
        

def score_lists(lists, dtm, label_var="LABEL"):
	labels = OrderedDict.fromkeys(lists[label_var])
	list_counts = pd.DataFrame(index=dtm.index, columns=labels)
	for label in list_counts.columns:
		tkns = lists.loc[lists[label_var] == label, "TOKEN"]
		tkns = [token for token in tkns if token in dtm.columns]
		list_counts[label] = dtm[tkns].sum(axis=1)
	list_scores = doc_mean_thres(list_counts)
	return list_scores


def make_cmap(colors, position=None, bit=False):
	'''
	http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
	make_cmap takes a list of tuples which contain RGB values. The RGB
	values may either be in 8-bit [0 to 255] (in which bit must be set to
	True when called) or arithmetic [0 to 1] (default). make_cmap returns
	a cmap with equally spaced colors.
	Arrange your tuples so that the first color is the lowest value for the
	colorbar and the last is the highest.
	position contains values from 0 to 1 to dictate the location of each color.
	'''
	import matplotlib as mpl
	import numpy as np
	bit_rgb = np.linspace(0,1,256)
	if position == None:
		position = np.linspace(0,1,len(colors))
	else:
		if len(position) != len(colors):
			sys.exit("position length must be the same as colors")
		elif position[0] != 0 or position[-1] != 1:
			sys.exit("position must start with 0 and end with 1")
	if bit:
		for i in range(len(colors)):
			colors[i] = (bit_rgb[colors[i][0]],
						 bit_rgb[colors[i][1]],
						 bit_rgb[colors[i][2]])
	cdict = {'red':[], 'green':[], 'blue':[]}
	for pos, color in zip(position, colors):
		cdict['red'].append((pos, color[0], color[0]))
		cdict['green'].append((pos, color[1], color[1]))
		cdict['blue'].append((pos, color[2], color[2]))

	cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
	return cmap


def load_atlas():

	import numpy as np
	from nilearn import image

	cer = "../data/brain/atlases/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz"
	cor = "../data/brain/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
	sub = "../data/brain/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

	sub_del_dic = {1:0, 2:0, 3:0, 12:0, 13:0, 14:0}
	sub_lab_dic_L = {4:1, 5:2, 6:3, 7:4, 9:5, 10:6, 11:7, 8:8}
	sub_lab_dic_R = {15:1, 16:2, 17:3, 18:4, 19:5, 20:6, 21:7, 7:8}

	sub_mat_L = image.load_img(sub).get_data()[91:,:,:]
	sub_mat_R = image.load_img(sub).get_data()[:91,:,:]

	for old, new in sub_del_dic.items():
		sub_mat_L[sub_mat_L == old] = new
	for old, new in sub_lab_dic_L.items():
		sub_mat_L[sub_mat_L == old] = new
	sub_mat_L = sub_mat_L + 48
	sub_mat_L[sub_mat_L == 48] = 0

	for old, new in sub_del_dic.items():
		sub_mat_R[sub_mat_R == old] = new
	for old, new in sub_lab_dic_R.items():
		sub_mat_R[sub_mat_R == old] = new
	sub_mat_R = sub_mat_R + 48
	sub_mat_R[sub_mat_R == 48] = 0

	cor_mat_L = image.load_img(cor).get_data()[91:,:,:]
	cor_mat_R = image.load_img(cor).get_data()[:91,:,:]

	mat_L = np.add(sub_mat_L, cor_mat_L)
	mat_L[mat_L > 56] = 0
	mat_R = np.add(sub_mat_R, cor_mat_R)
	mat_R[mat_R > 56] = 0

	mat_R = mat_R + 57
	mat_R[mat_R > 113] = 0
	mat_R[mat_R < 58] = 0

	cer_mat_L = image.load_img(cer).get_data()[91:,:,:]
	cer_mat_R = image.load_img(cer).get_data()[:91,:,:]
	cer_mat_L[cer_mat_L > 0] = 57
	cer_mat_R[cer_mat_R > 0] = 114

	mat_L = np.add(mat_L, cer_mat_L)
	mat_L[mat_L > 57] = 0
	mat_R = np.add(mat_R, cer_mat_R)
	mat_R[mat_R > 114] = 0

	mat = np.concatenate((mat_R, mat_L), axis=0)
	atlas_image = image.new_img_like(sub, mat)

	return atlas_image


def map_plane(estimates, atlas, path, suffix="", plane="z", cbar=False, annotate=False,
			  vmin=None, vmax=None, cmaps=[], print_fig=True, verbose=False):
	
	from nilearn import image, plotting
	from PIL import Image

	for f, feature in enumerate(estimates.columns):
		stat_map = image.copy_img(atlas).get_data()
		data = estimates[feature]
		if verbose:
			print("{:20s} Min: {:6.4f}  Mean: {:6.4f}  Max: {:6.4f}".format(
				  feature, min(data), np.mean(data), max(data)))
		if not verbose and print_fig:
			print("\n{}".format(feature))
		for i, value in enumerate(data):
			stat_map[stat_map == i+1] = value
		stat_map = image.new_img_like(atlas, stat_map)
		display = plotting.plot_stat_map(stat_map,
										 display_mode=plane, 
										 symmetric_cbar=False, colorbar=cbar,
										 cmap=cmaps[f], threshold=vmin, 
										 vmax=vmax, alpha=0.5,
										 annotate=annotate, draw_cross=False)
		file_name = "{}/{}{}.png".format(path, feature, suffix)
		display.savefig(file_name, dpi=250)

		# Remove image background
		img = Image.open(file_name)
		img = img.convert("RGBA")
		data = img.getdata()
		newData = []
		for item in data:
			if item[0] == 255 and item[1] == 255 and item[2] == 255:
				newData.append((255, 255, 255, 0))
			else:
				newData.append(item)
		img.putdata(newData)
		img.save(file_name, "PNG")

		if print_fig:
			plotting.show()
		display.close()
