#!/usr/bin/python

def make_cmap(colors, position=None, bit=False, n_colors=256):
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

	cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, n_colors)
	return cmap

def load_atlas():

	import pandas as pd
	import numpy as np
	from nilearn import image, plotting

	cer = "../data/atlases/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz"
	cor = "../data/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
	sub = "../data/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

	bilateral_labels = pd.read_csv("../data/atlases/harvard-oxford_orig.csv", index_col=0, header=0)

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