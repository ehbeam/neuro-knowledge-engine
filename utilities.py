#!/usr/bin/python3

import pandas as pd
import numpy as np
np.random.seed(42)


names = {"data-driven": "Data-Driven", "rdoc": "RDoC", "dsm": "DSM"}
suffix = {"data-driven": "", "rdoc": "_opsim", "dsm": "_opsim"}

c = {"red": "#CE7D69", "orange": "#BA7E39", "yellow": "#CEBE6D", 
	 "chartreuse": "#AEC87C", "green": "#77B58A", "blue": "#7597D0", 
	 "magenta": "#B07EB6", "purple": "#7D74A3", "brown": "#846B43", "pink": "#CF7593"}

palettes = {"data-driven": [c["blue"], c["magenta"], c["yellow"], c["green"], c["red"], c["purple"], c["chartreuse"], c["orange"], c["pink"], c["brown"]],
			"rdoc": [c["blue"], c["red"], c["green"], c["purple"], c["yellow"], c["orange"]],
			"dsm": [c["purple"], c["chartreuse"], c["orange"], c["blue"], c["red"], c["magenta"], c["yellow"], c["green"], c["brown"]]}

arial = "../style/Arial Unicode.ttf"


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_coordinates(path="../data"):
	atlas_labels = pd.read_csv("{}/brain/labels.csv".format(path))
	activations = pd.read_csv("{}/brain/coordinates.csv".format(path), index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]]
	return activations


def load_lexicon(sources, path="../lexicon"):
	lexicon = []
	for source in sources:
		file = "{}/lexicon_{}.txt".format(path, source)
		lexicon += [token.strip() for token in open(file, "r").readlines()]
	return sorted(lexicon)


def load_doc_term_matrix(version=190124, binarize=True, path="../data"):
	dtm = pd.read_csv("{}/text/dtm_{}.csv.gz".format(path, version), compression="gzip", index_col=0)
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm


def load_framework(framework, suffix="", circuit_suffix="", path="../ontology"):
	list_file = "{}/lists/lists_{}{}{}.csv".format(path, framework, suffix, circuit_suffix)
	lists = pd.read_csv(list_file, index_col=None)
	circuit_file = "{}/circuits/circuits_{}{}.csv".format(path, framework, circuit_suffix)
	circuits = pd.read_csv(circuit_file, index_col=0)
	return lists, circuits


def score_lists(lists, dtm_bin, label_var="DOMAIN"):
	from collections import OrderedDict
	labels = OrderedDict.fromkeys(lists[label_var])
	list_counts = pd.DataFrame(index=dtm_bin.index, columns=labels)
	for label in list_counts.columns:
		tkns = lists.loc[lists[label_var] == label, "TOKEN"]
		list_counts[label] = dtm_bin[tkns].sum(axis=1)
	list_scores = doc_mean_thres(list_counts)
	return list_scores


def transparent_background(file_name):
	from PIL import Image
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


def make_cmap(colors, position=None, bit=False):
	# Adapted from http://schubert.atmos.colostate.edu/~cslocum/custom_cmap.html
	
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

	cmap = mpl.colors.LinearSegmentedColormap("my_colormap",cdict,256)
	return cmap


cmaps = {"Yellows": make_cmap([(1,1,1), (0.937,0.749,0)]), 
		 "Magentas": make_cmap([(1,1,1), (0.620,0,0.686)]), 
		 "Purples": make_cmap([(1,1,1), (0.365,0,0.878)]),
		 "Chartreuses": make_cmap([(1,1,1), (0.345,0.769,0)]),
		 "Browns": make_cmap([(1,1,1), (0.82,0.502,0)])}


def load_atlas(path="../data"):

	import numpy as np
	from nilearn import image

	cer = "{}/brain/atlases/Cerebellum-MNIfnirt-maxprob-thr25-1mm.nii.gz".format(path)
	cor = "{}/brain/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz".format(path)
	sub = "{}/brain/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz".format(path)

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
		transparent_background(file_name)
		if print_fig:
			plotting.show()
		display.close()


def compare_to_null(df_null, df, domains, n_iter, alpha=0.001):
	from statsmodels.stats.multitest import multipletests
	pval = []
	for dom in domains:
		dom_null = df_null.loc[dom].values
		dom_obs = float(df.loc[dom, "OBSERVED"])
		p = np.sum(dom_null >= dom_obs) / float(n_iter)
		pval.append(p)
		df.loc[dom, "P"] = p
	df["FDR"] = multipletests(pval, method="fdr_bh")[1]
	for dom in domains:
		if df.loc[dom, "FDR"] < alpha:
			df.loc[dom, "STARS"] = "*"
		else:
			df.loc[dom, "STARS"] = ""
	df = df.loc[domains, ["OBSERVED", "P", "FDR", "STARS"]]
	return df
