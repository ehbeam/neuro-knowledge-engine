#!/usr/bin/python

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.stats.stats import pearsonr
from collections import OrderedDict
from nilearn import image, plotting
from sklearn.manifold import TSNE

def load_tokens(file, filter=[]):
	tokens = [token.strip() for token in open(file).readlines()]
	if len(filter) > 0:
		tokens = [token for token in tokens if token in filter]
	return sorted(tokens)

def load_activations(metric):
	atlas_labels = pd.read_csv("data/atlases/harvard-oxford.csv")
	activations = pd.read_csv("data/dcm_0mm_{}.csv".format(metric), index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]
	return activations

def load_atlas():

	cer = "data/atlases/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz"
	cor = "data/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
	sub = "data/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

	bilateral_labels = pd.read_csv("data/atlases/harvard-oxford_orig.csv", index_col=0, header=0)

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

def map_stats(estimates, path, features=[], vmax=None, print_fig=True):
	atlas_image = load_atlas()
	plt.rcParams["font.size"] = 12
	if len(features) == 0:
		features = estimates.columns
	for feature in features:
		stat_map = image.copy_img(atlas_image).get_data()
		for i, value in enumerate(estimates[feature]):
			stat_map[stat_map == i+1] = value
		stat_map = image.new_img_like(atlas_image, stat_map)
		if not vmax:
			vmax = max([np.max(np.max(estimates)), abs(np.min(np.min(estimates)))])
		display = plotting.plot_stat_map(stat_map,
										 cmap="RdBu_r",
										 symmetric_cbar=True,
										 vmax=vmax,
										 alpha=0.75,
										 cut_coords=(-4,-16,9),
										 annotate=True,
										 black_bg=False,
										 draw_cross=False)
		display.savefig("{}/{}.png".format(path, feature), dpi=250)
		if print_fig:
			print("\n" + feature)
			plotting.show()
		display.close()

def reconcile_indices(X, Y):
	idx = set(X.index).intersection(set(Y.index))
	X = X.loc[idx].dropna().sort_index()
	Y = Y.loc[idx].dropna().sort_index()
	return X, Y

def correlate(X, Y, reconcile_XY=True, p_thres=1e-5):
	if reconcile_XY:
		X, Y = reconcile_indices(X, Y)
	R = np.empty([len(X.columns), len(Y.columns)])
	P = np.empty([len(X.columns), len(Y.columns)])
	for i, x in enumerate(X):
		for j, y in enumerate(Y):
			R[i,j], P[i,j] = pearsonr(X[x], Y[y])
	R[P > p_thres] = 0
	R = pd.DataFrame(R, index=X.columns, columns=Y.columns)
	return R

def average_embeddings(vsm, labels, tokens):
	label_set = list(OrderedDict.fromkeys(labels))
	centroids = pd.DataFrame(index=label_set, columns=range(vsm.shape[1]))
	for label in label_set:
		idx = [i for i, l in enumerate(labels) if l == label]
		centroid = np.average(vsm.loc[tokens[min(idx):(max(idx)+1)]], axis=0)
		centroids.loc[label,:] = centroid
	return centroids

def load_columns(orig_df, new_df, columns, orig_label="CONSTRUCT", new_label="CONSTRUCT"):
	for col in columns:
		data = []
		for i, row in new_df.iterrows():
			data.append(list(orig_df.loc[orig_df[orig_label] == row[new_label], col])[0])
		new_df[col] = data
	return new_df