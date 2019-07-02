#!/usr/bin/python3

import argparse

parser = argparse.ArgumentParser(description="Generate plots for 'A computational knowledge engine for human neuroscience'")
parser.add_argument("clf", type=str, default="lr", options=["lr", "nn"], 
					help="Classification architecture ('lr' for logistic regression, 'nn' for neural network)")
args = parser.parse_args()


################################################
############### 0. Load the data ###############
################################################

print("\n--- Loading the data ---")

import utilities

vsm_version = 190428 # Version of GloVe embeddings
dtm_version = 190325 # Version of document-term matrix
rdoc_version = 190124 # Version of RDoC matrix

frameworks = ["data-driven", "rdoc", "dsm"] # Frameworks to analyze
titles = {"data-driven": "Data-Driven", "rdoc": "RDoC", "dsm": "DSM"}
suffixes = {"data-driven": "_lr", "rdoc": "_opsim", "dsm": "_opsim"} # For framework infiles 

# Text and activation coordinate inputs
dtm = utilities.load_doc_term_matrix(version=dtm_version, binarize=True, path="data") # Document-term matrix
act = utilities.load_coordinates(path="data") # Activation coordinates

# Atlas for brain circuit plots
atlas = utilities.load_atlas(path="data")

# PMIDs and their splits
pmids = act.index.intersection(dtm.index)
splits = {split: [int(pmid.strip()) for pmid in open("data/splits/{}.txt".format(split))] 
				  for split in ["train", "validation", "test"]}

# Iterations for bootstrap and null distributions
n_iter = 1000

# Font for plots
arial = "style/Arial Unicode.ttf"


################################################
##### 1. Generate the data-driven ontology #####
################################################

print("\n--- Generating the data-driven ontology ---")

import os
import pickle
import pandas as pd
from ontology import ontology
from sklearn.cluster import KMeans
from sklearn.preprocessing import binarize

# Load the lexicon
lexicon = utilities.load_lexicon(["cogneuro"], path="lexicon", tkn_filter=dtm.columns)

# Compute the PMI-weighted structure-term matrix
stm = ontology.load_stm(act.loc[splits["train"]], dtm.loc[splits["train"], lexicon]) 

# Specify the output ranges for number of circuits and terms per circuit
circuit_counts = range(2, 26) # Range over which ROC-AUC becomes asymptotic
list_lens = range(5, 26) # Same range as RDoC and the DSM

# Generate the data-driven domains over the specified ranges
for k in circuit_counts:

	# Cluster structures by functions
	circuit_file = "ontology/circuits/circuits_k{:02d}.csv".format(k)
	if not os.path.isfile(circuit_file):
		clust = ontology.cluster_structures(k, stm, act.columns)
		clust.to_csv(circuit_file, index=None)

	# Associate functions to circuits
	list_file = "ontology/lists/lists_k{:02d}.csv".format(k)
	if not os.path.isfile(list_file):
		lists = ontology.assign_functions(clust, splits, act, dtm, list_lens=list_lens)
		lists.to_csv(list_file, index=None)

# Select optimal number of words per domain
# Note: Run on Sherlock, generating ontology/lists/*_oplen.csv

# Select optimal number of domains

directions = ["forward", "reverse"]

fits = {}
for direction in directions:
	fits[direction] = {}
	for k in circuit_counts:
		state_dict = torch.load("ontology/sherlock/fits/{}_k{:02d}.pt".format(direction, k))
		hyperparams = pd.read_csv("ontology/data/params_data-driven_k{:02d}_{}.csv".format(k, direction), 
								  header=None, index_col=0)
		h = {str(label): float(value) for label, value in hyperparams.iterrows()}
		layers = list(state_dict.keys())
		n_input = state_dict[layers[0]].shape[1]
		n_output = state_dict[layers[-2]].shape[0]
		fits[direction][k] = Net(n_input=n_input, n_output=n_output, 
						 n_hid=int(h["n_hid"]), p_dropout=h["p_dropout"])
		fits[direction][k].load_state_dict(state_dict)

features = {k: {} for k in circuit_counts}
for k in circuit_counts:
	domains = range(1, k+1)
	lists, circuits = ontology.load_ontology(k, path="ontology/")
	function_features = pd.DataFrame(index=dtm.index, columns=domains)
	structure_features = pd.DataFrame(index=act.index, columns=domains)
	for i in domains:
		functions = lists.loc[lists["CLUSTER"] == i, "TOKEN"]
		function_features[i] = dtm[functions].sum(axis=1)
		structures = circuits.loc[circuits["CLUSTER"] == i, "STRUCTURE"]
		structure_features[i] = act[structures].sum(axis=1)
	function_features = pd.DataFrame(utilities.doc_mean_thres(function_features), 
									 index=dtm.index, columns=domains)
	structure_features = pd.DataFrame(binarize(structure_features), 
									 index=act.index, columns=domains)
	features[k]["function"] = function_features
	features[k]["structure"] = structure_features


rocauc_scores = ontology.compute_eval_scores(roc_auc_score, directions, 
											 circuit_counts, features, fits, val)
rocauc_boot = ontology.compute_eval_boot(roc_auc_score, directions, circuit_counts, features, 
										 fits, val, n_iter=n_iter, verbose=True, path="ontology/")
rocauc_null = ontology.compute_eval_null(roc_auc_score, directions, circuit_counts, features, 
										 fits, val, n_iter=n_iter, verbose=True, path="ontology/")

op_k = 6 
for direction, shape in zip(directions + ["mean"], [">", "<", "D"]):
	ontology.plot_scores(rocauc_scores, direction, circuit_counts, rocauc_boot, rocauc_null, 
						 "rocauc", shape=shape, ylim=[0.45,0.65], yticks=[0.45,0.5,0.55,0.6,0.65],
						 op_k=op_k, interval=0.95, font=arial, path="ontology/", print_fig=False)
lists, circuits = ontology.load_ontology(op_k, path="ontology/")

# Name the domains
k2name = {}
for k in range(op_k):
	degrees = ontology.term_degree_centrality(k+1, lists, circuits, dtm, splits["train"])
	name = degrees.index[0].upper()
	k2name[k+1] = name

# Reorder the domains
order = [3,6,5,4,2,1]
names = [k2name[k] for k in order]
k2order = {k: order.index(k)+1 for k in range(1,op_k+1)}

# Export the named term lists
lists["ORDER"] = [k2order[k] for k in lists["CLUSTER"]]
lists["DOMAIN"] = [k2name[k] for k in lists["CLUSTER"]]
lists = lists.sort_values(["ORDER", "R"], ascending=[True, False])
lists = lists[["ORDER", "CLUSTER", "DOMAIN", "TOKEN", "R", "ROC_AUC"]]
lists.to_csv("ontology/lists/lists_data-driven.csv", index=None)

# Plot the term lists
print("\nPlotting word clouds")
ontology.plot_wordclouds("data-driven", names, lists, dtm[lexicon], width=600,
						 path="ontology/", font_path=arial, print_fig=False)

# Export the named circuits
circuits["ORDER"] = [k2order[k] for k in circuits["CLUSTER"]]
circuits["DOMAIN"] = [k2name[k] for k in circuits["CLUSTER"]]
circuits = circuits.sort_values(["ORDER", "STRUCTURE"])
circuits = circuits[["ORDER", "CLUSTER", "DOMAIN", "STRUCTURE"]]
circuits.to_csv("ontology/circuits/clusters_data-driven.csv", index=None)

# Plot the named circuits
print("\nPlotting circuit maps")
circuit_mat = pd.DataFrame(0.0, index=act.columns, columns=names)
for name in names:
	structures = circuits.loc[circuits["DOMAIN"] == name, "STRUCTURE"]
	for structure in structures:
		circuit_mat.loc[structure, name] = 1.0
circuit_mat.to_csv("ontology/circuits/circuits_data-driven.csv")
utilities.map_plane(circuit_mat, atlas, "ontology/figures/circuits/data-driven", 
		  			suffix="_z", plane="z", cmaps=utilities.colormaps["data-driven"], cbar=True, vmin=0.0, vmax=2.0,
		  			verbose=False, print_fig=False, annotate=True)


################################################
###### 2. Generate the expert frameworks #######
################################################

import collections
import numpy as np
np.random.seed(42)

n_iter_fw = 10000 # Iterations for seed similarity and PPMI circuits

list_lens = range(5, 26) # Range of word list lengths
list_len = list(list_lens)[-1]

alphas = [0.01, 0.001] # Statistical significance level for seed similarity
interval = 0.95 # Confidence interval for seed similarity null distribution

dtm_fw = utilities.load_doc_term_matrix(version=dtm_version, binarize=False, path="data")
dtm_fw = dtm_fw.loc[:, (dtm_fw != 0).any(axis=0)]
dtm_fw = utilities.doc_mean_thres(dtm_fw)


###### RDoC ######
print("\n--- Generating the rdoc framework ---")

vsm = pd.read_csv("data/text/glove_gen_n100_win15_min5_iter500_{}.txt".format(vsm_version), 
					index_col=0, header=None, sep=" ")

seed_df = pd.read_csv("lexicon/rdoc_{}/rdoc_seeds.csv".format(rdoc_version), 
					  index_col=None, header=0)
seed_df = seed_df.loc[seed_df["TOKEN"].isin(vsm.index)]
doms = list(collections.OrderedDict.fromkeys(seed_df["DOMAIN"]))

lexicon = sorted(list(set(lexicon).union(seed_df["TOKEN"]).intersection(dtm_fw.columns).intersection(vsm.index)))

lists = ontology.load_rdoc_lists(lexicon, vsm, seed_df, n_thres=list_len)
lists = lists.loc[lists["TOKEN"].isin(dtm_fw.columns)]
lists.to_csv("ontology/lists/lists_rdoc.csv", index=None)

op_df = ontology.load_optimized_lists(doms, lists, list_lens, seed_df, vsm)
op_df.to_csv("ontology/data/df_rdoc_opsim.csv")
new = ontology.update_lists(doms, op_df, lists, "rdoc")

# Compare RDoC to McCoy similarity to seeds
mccoy = pd.read_csv("ontology/lists/lists_cqh.csv", index_col=None)
seed_centroid = ontology.compute_centroid(seed_df, doms, vsm)
new_centroid = ontology.compute_centroid(new, doms, vsm)
mccoy_centroid = ontology.compute_centroid(mccoy, doms, vsm)

new_sim_boot, mccoy_sim_boot = np.empty((len(doms), n_iter_fw)), np.empty((len(doms)-1, n_iter_fw))
for n in range(n_iter_fw):
	new_sim_boot[:,n] = 1.0 - ontology.compute_sims_sample(new_centroid, seed_centroid, vsm)
	mccoy_sim_boot[:(len(doms)-1),n] = (1.0 - ontology.compute_sims_sample(mccoy_centroid, seed_centroid, vsm))[:(len(doms)-1)]

new_sim_null, mccoy_sim_null = np.empty((len(doms), n_iter_fw)), np.empty((len(doms), n_iter_fw))
for n in range(n_iter_fw):
	new_sim_null[:,n] = 1.0 - ontology.compute_sims_shuffle(new_centroid, seed_centroid, vsm)
	
new_sim_obs = np.reshape(1.0 - ontology.compute_sims(new, seed_centroid, doms, vsm), (len(doms), 1))
mccoy_sim_obs = np.reshape(1.0 - ontology.compute_sims(mccoy, seed_centroid, doms, vsm), (len(doms), 1))

pvals_dif = np.sum(np.less(new_sim_boot[:(len(doms)-1),:] - mccoy_sim_boot, 0.0), axis=1) / n_iter_fw
fdrs_dif = multipletests(pvals_dif, method="fdr_bh")[1] # Benjamini-Hochberg

lower = [sorted(new_sim_null[i,:])[int(n_iter_fw*(1.0-interval))] for i in range(len(doms))]
upper = [sorted(new_sim_null[i,:])[int(n_iter_fw*interval)] for i in range(len(doms))]

print("\nPlotting similarity to seeds")
ontology.plot_rdoc_similarity(doms, new_sim_null, lower, upper, mccoy_sim_boot, mccoy_sim_obs,
					 new_sim_boot, new_sim_obs, fdrs_dif, alphas, font=arial)

# Plot the term lists
print("\nPlotting word clouds")
ontology.plot_wordclouds("rdoc", doms, new, dtm, 
						 path="ontology/", font_path=arial, print_fig=False)

# Map the brain circuits
print("\nPlotting circuit maps")
dom_links_thres = ontology.load_framework_circuit(new, dtm, act, "rdoc")
utilities.map_plane(dom_links_thres, atlas, "ontology/figures/circuits/rdoc", suffix="_z", 
	  				cmaps=utilities.colormaps["rdoc"], plane="z", cbar=True, vmin=0.0, vmax=0.6,
	  				verbose=False, print_fig=False, annotate=True)


###### DSM ######
print("\n--- Generating the dsm framework ---")

vsm = pd.read_csv("data/text/glove_psy_n100_win15_min5_iter500_{}.txt".format(vsm_version), 
				  index_col=0, header=None, sep=" ")

seed_df = pd.read_csv("data/text/seeds_dsm5.csv", index_col=None, header=0)
doms = list(collections.OrderedDict.fromkeys(seed_df["DOMAIN"]))

lexicon = utilities.load_lexicon(["cogneuro", "dsm", "psychiatry"], path="lexicon")
lexicon = sorted(list(set(lexicon).intersection(vsm.index).intersection(dtm_fw.columns)))

class_tkns = []
for dom in doms:
	class_tkns += set(seed_df.loc[seed_df["DOMAIN"] == dom, "TOKEN"])
unique = [tkn for tkn in class_tkns if class_tkns.count(tkn) == 1]

lists = ontology.load_dsm_lists(lexicon, vsm, seed_df, unique, n_thres=list_len, verbose=True)
lists.to_csv("ontology/lists/lists_dsm.csv", index=None)

op_df = ontology.load_optimized_lists(doms, lists, list_lens, seed_df, vsm)
new = ontology.update_lists(doms, op_df, lists, "dsm")

# Threshold domains by those with one or more terms in >5% of articles with coordinate data
doms = list(collections.OrderedDict.fromkeys(seed_df["DOMAIN"]))
filt_doms = []
for dom in doms: 
	tkns = set(new.loc[new["DOMAIN"] == dom, "TOKEN"])
	freq = sum([1.0 for doc in dtm_fw[tkns].sum(axis=1) if doc > 0]) / float(len(dtm_fw))
	if freq > 0.05:
		filt_doms.append(dom)
doms = filt_doms

new = new.loc[new["DOMAIN"].isin(filt_doms)]
new = new.loc[new["DISTANCE"] > 0]
op_df.to_csv("ontology/data/df_dsm_opsim.csv")
new.to_csv("ontology/lists/lists_dsm_opsim.csv", index=None)

# Plot the term lists
print("\nPlotting word clouds")
ontology.plot_wordclouds("dsm", doms, new, dtm,
						 path="ontology/", font_path=arial, print_fig=False)

# Map the brain circuits
print("\nPlotting circuit maps")
dom_links_thres = ontology.load_framework_circuit(new, dtm, act, "dsm")
utilities.map_plane(dom_links_thres, atlas, "ontology/figures/circuits/dsm", suffix="_z", 
	  				cmaps=utilities.colormaps["dsm"], plane="z", cbar=True, vmin=0.0, vmax=0.6,
	  				verbose=False, print_fig=False, annotate=True)


################################################
########## 3. Assess reproducibility ###########
################################################

print("\n--- Assessing reproducibility ---")

from prediction import evaluation
from prediction.evaluation import Net
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
torch.manual_seed(42)

opt_epochs = 500 # Epochs used to optimize the classifier hyperparameters
train_epochs = 1000 # Epochs used to train the classifier	
directions = ["forward", "reverse"] # Directions for classifier inference
metric_labels = ["rocauc", "f1"] # Evaluation metrics to be computed in the test set
alpha = 0.001 # Significance level for plotting statistical comparisons
interval = 0.999 # Confidence interval for plotting null distributions

# Brain structure labels
struct_labels = pd.read_csv("data/brain/labels.csv", index_col=None)
struct_labels.index = struct_labels["PREPROCESSED"]
struct_labels = struct_labels.loc[act.columns, "ABBREVIATION"].values

# Parameters for plotting evaluation metrics
axis_labels = {"forward": struct_labels, "reverse": []}
figsize = {"forward": (13, 3.2), "reverse": (3.6, 3.2)}
ylim = {"rocauc": [0.4, 0.8], "f1": [0.3, 0.7]}
dxs = {"forward": {"data-driven": 0.55, "rdoc": 0.55, "dsm": 0.55}, 
	   "reverse": {"data-driven": 0.11, "rdoc": 0.11, "dsm": 0.17}}
opacity = {"forward": 0.4, "reverse": 0.65}

obs, boot, null = {}, {}, {} # Dictionaries for framework-level results

for framework in frameworks:

	print("\n-- Processing {} framework --".format(framework))

	lists, circuits = utilities.load_framework(framework, suffix=suffixes[framework], path="ontology")
	scores = utilities.score_lists(lists, dtm)
	domains = list(circuits.columns)

	fit = {}
	for direction in directions:
		hyperparams = pd.read_csv("prediction/data/params_{}_{}_{}epochs.csv".format(framework, direction, opt_epochs), header=None, index_col=0)
		h = {str(label): float(value) for label, value in hyperparams.iterrows()}
		state_dict = torch.load("prediction/fits/{}_{}_{}epochs.pt".format(framework, direction, train_epochs))
		layers = list(state_dict.keys())
		n_input = state_dict[layers[0]].shape[1]
		n_output = state_dict[layers[-2]].shape[0]
		fit[direction] = Net(n_input=n_input, n_output=n_output, 
							 n_hid=int(h["n_hid"]), p_dropout=h["p_dropout"])
		fit[direction].load_state_dict(state_dict)

	test_set = evaluation.load_mini_batches(scores, act, splits["test"], mini_batch_size=len(splits["test"]), seed=42)[0]
	test_set = evaluation.numpy2torch(test_set)
	scores_tensor, act_tensor = test_set

	palette = {"forward": [], "reverse": utilities.palettes[framework]}
	for structure in act.columns:
		dom_idx = np.argmax(circuits.loc[structure].values)
		color = palette["reverse"][dom_idx]
		palette["forward"].append(color)

	for direction, features, labels in zip(directions, [scores_tensor, act_tensor], [act_tensor, scores_tensor]):
		with torch.no_grad():
			pred_probs = fit[direction](features).numpy()

		fpr, tpr = evaluation.compute_roc(labels, pred_probs)
		evaluation.plot_curves("roc", framework, direction, fpr, tpr, palette[direction], 
							   opacity=opacity[direction], font=arial, path="prediction/", print_fig=False)

		precision, recall = evaluation.compute_prc(labels, pred_probs)
		evaluation.plot_curves("prc", framework, direction, recall, precision, palette[direction], 
							   opacity=opacity[direction], diag=False, font=arial, path="prediction/", print_fig=False)

	X = {"forward": scores_tensor, "reverse": act_tensor}
	Y = {"forward": act_tensor, "reverse": scores_tensor}
	with torch.no_grad():
		pred_probs = {direction: fit[direction](X[direction]).numpy() for direction in directions}
	preds = {direction: 1 * (pred_probs[direction] > 0.5) for direction in directions}

	obs[framework] = {"name": titles[framework]}
	for direction in directions:
		obs[framework][direction] = {}
		obs[framework][direction]["rocauc"] = evaluation.compute_eval_metric(Y[direction], pred_probs[direction], roc_auc_score)
		obs[framework][direction]["f1"] = evaluation.compute_eval_metric(Y[direction], preds[direction], f1_score)

	boot[framework] = {"name": titles[framework]} 
	for direction in directions:
		boot[framework][direction] = {}
		boot[framework][direction]["rocauc"] = np.empty((len(obs[framework][direction]["rocauc"]), n_iter))
		boot[framework][direction]["f1"] = np.empty((len(obs[framework][direction]["f1"]), n_iter))
		
		rocauc_file = "prediction/data/rocauc_boot_{}_{}_{}iter.csv".format(framework, direction, n_iter)
		if os.path.isfile(rocauc_file):
			boot[framework][direction]["rocauc"] = pd.read_csv(rocauc_file, index_col=0, header=0).values
		else:
			for n in range(n_iter):
				samp = np.random.choice(range(len(splits["test"])), size=len(splits["test"]), replace=True)
				boot[framework][direction]["rocauc"][:,n] = evaluation.compute_eval_metric(Y[direction][samp,:], pred_probs[direction][samp,:], roc_auc_score)
		
		f1_file = "prediction/data/f1_boot_{}_{}_{}iter.csv".format(framework, direction, n_iter)
		if os.path.isfile(f1_file):
			boot[framework][direction]["f1"] = pd.read_csv(f1_file, index_col=0, header=0).values
		else:
			for n in range(n_iter):
				samp = np.random.choice(range(len(splits["test"])), size=len(splits["test"]), replace=True)
				boot[framework][direction]["f1"][:,n] = evaluation.compute_eval_metric(Y[direction][samp,:], preds[direction][samp,:], f1_score)

	null[framework] = {"name": titles[framework]} 
	for direction in directions:
		print("\n   {}".format(direction.upper()))
		null[framework][direction] = {}
		null[framework][direction]["rocauc"] = np.empty((len(obs[framework][direction]["rocauc"]), n_iter))
		null[framework][direction]["f1"] = np.empty((len(obs[framework][direction]["f1"]), n_iter))
		
		rocauc_file = "prediction/data/rocauc_null_{}_{}_{}iter.csv".format(framework, direction, n_iter)
		if os.path.isfile(rocauc_file):
			null[framework][direction]["rocauc"] = pd.read_csv(rocauc_file, index_col=0, header=0).values
		else:
			for n in range(n_iter):
				shuf = np.random.choice(range(len(splits["test"])), size=len(splits["test"]), replace=False)
				null[framework][direction]["rocauc"][:,n] = evaluation.compute_eval_metric(Y[direction][shuf,:], pred_probs[direction], roc_auc_score)
				if n % (n_iter/10) == 0:
					print("\tProcessed {} iterations".format(n))
		
		f1_file = "prediction/data/f1_null_{}_{}_{}iter.csv".format(framework, direction, n_iter)
		if os.path.isfile(f1_file):
			null[framework][direction]["f1"] = pd.read_csv(f1_file, index_col=0, header=0).values
		else:
			for n in range(n_iter):
				samp = np.random.choice(range(len(splits["test"])), size=len(splits["test"]), replace=True)
				null[framework][direction]["f1"][:,n] = evaluation.compute_eval_metric(Y[direction][shuf,:], preds[direction], f1_score)

	idx_lower = int((1.0-interval)*n_iter)
	idx_upper = int(interval*n_iter)
	null_ci = {direction: {} for direction in directions}
	for metric in metric_labels:
		for direction in directions:
			dist = null[framework][direction][metric]
			n_clf = dist.shape[0]
			null_ci[direction][metric] = {}
			null_ci[direction][metric]["lower"] = [sorted(dist[i,:])[idx_lower] for i in range(n_clf)]
			null_ci[direction][metric]["upper"] = [sorted(dist[i,:])[idx_upper] for i in range(n_clf)]
			null_ci[direction][metric]["mean"] = [np.mean(dist[i,:]) for i in range(n_clf)]

	p = {direction: {} for direction in directions}
	for metric in metric_labels:
		for direction in directions:
			dist = null[framework][direction][metric]
			n_clf = dist.shape[0]
			p[direction][metric] = [np.sum(dist[i,:] >= obs[framework][direction][metric][i]) / float(n_iter) for i in range(n_clf)]

	fdr = {direction: {} for direction in directions}
	for metric in metric_labels:
		for direction in directions:
			fdr[direction][metric] = multipletests(p[direction][metric], method="fdr_bh")[1]

	labels = {"forward": act.columns, "reverse": domains}
	for metric in metric_labels:
		for direction in directions:
			for dist, dic in zip(["boot", "null"], [boot, null]):
				df = pd.DataFrame(dic[framework][direction][metric], 
								  index=labels[direction], columns=range(n_iter))
				df.to_csv("prediction/data/{}_{}_{}_{}_{}iter.csv".format(
						  metric, dist, framework, direction, n_iter))
			obs_df = pd.Series(obs[framework][direction][metric], index=labels[direction])
			obs_df.to_csv("prediction/data/{}_obs_{}_{}.csv".format(metric, framework, direction))

	for direction in directions:
		for metric in metric_labels:
			evaluation.plot_eval_metric(metric, framework, direction, obs[framework][direction][metric], 
							 boot[framework][direction][metric], null_ci[direction][metric], fdr[direction][metric],
							 palette[direction], labels=axis_labels[direction], dx=0.375, dxs=dxs[direction][framework], print_fig=False,
							 figsize=figsize[direction], ylim=ylim[metric], alpha=alpha, font=arial, path="prediction/")

# Compare the frameworks
fdr, sig = {}, {}
for metric in metric_labels:
	fdr[metric], sig[metric] = {}, {}
	for direction in directions:
		p = np.empty((len(frameworks), len(frameworks)))
		for i, fw_i in enumerate(frameworks):
			for j, fw_j in enumerate(frameworks):
				boot_i = np.mean(boot[fw_i][direction][metric], axis=0)
				boot_j = np.mean(boot[fw_j][direction][metric], axis=0)
				p[i,j] = np.sum((boot_i - boot_j) <= 0.0) / float(n_iter)
		
		fdr_md = multipletests(p.ravel(), method="fdr_bh")[1].reshape(p.shape)
		fdr_md = pd.DataFrame(fdr_md, index=frameworks, columns=frameworks)
		fdr[metric][direction] = fdr_md
		
		sig_md = fdr_md.copy()
		sig_md[fdr_md >= alpha] = ""
		sig_md[fdr_md < alpha] = "*"
		sig[metric][direction] = sig_md

for direction in directions:
	print("-- {} --".format(direction.upper()))
	print("\nROC-AUC")
	print(fdr["rocauc"][direction])
	evaluation.plot_framework_comparison("rocauc", direction, boot, n_iter=n_iter, ylim=[0.4,0.8], font=arial,
							  			 yticks=[0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8], path="prediction/", print_fig=False)
	print("\nF1")
	print(fdr["f1"][direction])
	evaluation.plot_framework_comparison("f1", direction, boot, n_iter=n_iter, ylim=[0.3,0.7], font=arial,
							  			 yticks=[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7], path="prediction/", print_fig=False)
	print("")

dif_thres = {}
for framework in ["rdoc", "dsm"]:
	dif_thres[framework] = {}
	for metric in metric_labels:
		dif_null = null["data-driven"]["forward"][metric] - null[framework]["forward"][metric]
		dif_obs = np.array(obs["data-driven"]["forward"][metric]) - np.array(obs[framework]["forward"][metric])
		pvals = [np.sum(np.abs(dif_null[i,:]) > np.abs(dif_obs[i])) / n_iter for i in range(len(dif_obs))]
		fdrs = multipletests(pvals, method="fdr_bh")[1]
		dif_obs[fdrs >= alpha] = np.nan
		dif_thres[framework][metric] = pd.DataFrame(dif_obs)
		dif_thres[framework][metric].columns = [""]

print("\nMapping differences in forward inference")
for framework in ["rdoc", "dsm"]:
	for metric in ["rocauc", "f1"]:
		utilities.map_plane(dif_thres[framework][metric], atlas, "prediction/figures", 
				  			suffix="{}_dif_{}_{}iter".format(metric, framework, n_iter), 
				  			plane="ortho", cbar=True, annotate=True, vmin=0, vmax=0.2, 
				  			cmaps=["RdBu_r"], print_fig=False, verbose=False)


################################################
### 4. Assess modularity & generalizability ####
################################################

print("\n--- Assessing modularity & generalizability --")

from archetype import archetype
from modularity import modularity
from scipy.spatial.distance import dice, cdist

# Nudges for plotted means
mod_dx = {"data-driven": [0.36, 0.35, 0.38, 0.38, 0.34, 0.35],
	  	  "rdoc": [0.31, 0.38, 0.37, 0.40, 0.36, 0.38],
	  	  "dsm": [0.38, 0.38, 0.35, 0.38, 0.37, 0.38, 0.37, 0.36, 0.32]}
gen_dx = {"data-driven": [0.39, 0.38, 0.37, 0.39, 0.39, 0.38],
	  	  "rdoc": [0.31, 0.4, 0.36, 0.39, 0.39, 0.37],
	  	  "dsm": [0.36, 0.37, 0.39, 0.39, 0.39, 0.39, 0.32, 0.34, 0.39]}

# Nudges for plotted stars
mod_ds = {"data-driven": 0.09, "rdoc": 0.09, "dsm": 0.13}
gen_ds = {"data-driven": 0.11, "rdoc": 0.09, "dsm": 0.13}

# Significance level for statistical comparison tests
alpha = 0.001

for framework in frameworks:

	print("\n-- Processing {} framework --".format(framework))

	# Load framework data
	lists, circuits = utilities.load_framework(framework, suffix=suffixes[framework], path="ontology")
	words = sorted(list(set(lists["TOKEN"])))
	structures = sorted(list(set(act.columns)))
	domains = list(collections.OrderedDict.fromkeys(lists["DOMAIN"]))

	# Compute "archetypes" of included words and structures
	archetypes = pd.DataFrame(0.0, index=words+structures, columns=domains)
	for dom in domains:
		for word in lists.loc[lists["DOMAIN"] == dom, "TOKEN"]:
			archetypes.loc[word, dom] = 1.0
		for struct in structures:
			archetypes.loc[struct, dom] = circuits.loc[struct, dom]
	archetypes[archetypes > 0.0] = 1.0
	dtm_words = dtm.loc[pmids, words]
	act_structs = act.loc[pmids, structures]
	docs = dtm_words.copy()
	docs[structures] = act_structs.copy()

	# Partition articles by similarity to domain archetypes
	print("   Partitioning articles")
	partition_file = "partition/data/doc2dom_{}.csv".format(framework)
	if not os.path.isfile(partition_file):

		dom_dists = cdist(docs.values, archetypes.values.T, metric="dice")

		doc2dom = {pmid: 0 for pmid in pmids}
		for i, pmid in enumerate(pmids):
			doc2dom[pmid] = np.argmin(dom_dists[i,:]) + 1

		doc2dom_df = pd.Series(doc2dom)
		doc2dom_df.to_csv(partition_file, header=False)

	else:
		doc2dom_df = pd.read_csv(partition_file, header=None, index_col=0)
		doc2dom = {int(pmid): int(dom) for pmid, dom in doc2dom_df.iterrows()}

	dom2docs = {dom: [] for dom in domains}
	for doc, dom in doc2dom.items():
		dom2docs[domains[dom-1]].append(doc)
	sorted_pmids = []
	for dom in range(len(domains)):
		sorted_pmids += [pmid for pmid, sys in doc2dom.items() if sys == dom + 1]
	dom_idx = {dom: {"min": 0, "max": 0} for dom in domains}
	for dom in domains:
		dom_pmids = dom2docs[dom]
		dom_idx[dom]["min"] = sorted_pmids.index(dom_pmids[0])
		dom_idx[dom]["max"] = sorted_pmids.index(dom_pmids[-1]) + 1

	### Modularity ###
	print("   Analyzing modularity")

	# Compute distances between articles
	doc_dists = cdist(docs, docs, metric="dice")
	doc_dists = pd.DataFrame(doc_dists, index=pmids, columns=pmids)
	doc_dists = doc_dists[sorted_pmids].loc[sorted_pmids]

	dists_int, dists_ext = {}, {}
	df_obs = pd.DataFrame(index=domains, columns=pmids)
	df = pd.DataFrame(index=domains, columns=["OBSERVED"])

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
		
		df.loc[dom, "OBSERVED"] = dist_ext / dist_int
		df_obs.loc[dom, dom2docs[dom]] = doc_dists_ext / doc_dists_int

	df.to_csv("modularity/data/mod_obs_{}.csv".format(framework))

	null_dists = doc_dists.values.copy()
	file_null = "modularity/data/mod_null_{}_{}iter.csv".format(framework, n_iter)
	if not os.path.isfile(file_null):
		print("\tComputing null distribution")
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

	file_boot = "modularity/data/mod_boot_{}_{}iter.csv".format(framework, n_iter)
	if not os.path.isfile(file_boot):
		print("\tComputing bootstrap distribution")
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

	stats = utilities.compare_to_null(df_null, df, domains, n_iter, alpha=alpha)

	modularity.plot_violins(framework, domains, stats, df_null, df_obs, utilities.palettes[framework], 
				 			dx=mod_dx[framework], ds=mod_ds[framework], alphas=[0], interval=0.999, print_fig=False,
				 			ylim=[0.75,1.75], yticks=[0.75,1,1.25,1.5,1.75], font=arial, path="modularity/")


	### Generalizability ###
	print("   Analyzing generalizability")

	df_obs = pd.DataFrame(index=domains, columns=pmids)
	for dom in domains:
		dom_pmids = dom2docs[dom]
		dom_vecs = docs.loc[dom_pmids].values
		dom_arche = archetypes[dom].values.reshape(1, archetypes.shape[0])
		dom_sims = 1.0 - cdist(dom_vecs, dom_arche, metric="dice")
		df_obs.loc[dom, dom_pmids] = dom_sims[:,0]
	df = pd.DataFrame({"OBSERVED": df_obs.mean(axis=1)}, index=domains)
	df.to_csv("archetype/data/arche_obs_{}.csv".format(framework))

	file_null = "archetype/data/arche_null_{}_{}iter.csv".format(framework, n_iter)
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
				df_null[i,n] = 1.0 - np.mean(cdist(dom_vecs, dom_arche, metric="dice"))
			if n % int(n_iter / 10.0) == 0:
				print("\t\tProcessed {} iterations".format(n))
		df_null = pd.DataFrame(df_null, index=domains, columns=range(n_iter))
		df_null.to_csv(file_null)
	else:
		df_null = pd.read_csv(file_null, index_col=0, header=0)

	file_boot = "archetype/data/arche_boot_{}_{}iter.csv".format(framework, n_iter)
	if not os.path.isfile(file_boot):
		print("\tComputing bootstrap distribution")
		df_boot = np.zeros((len(domains), n_iter))
		for n in range(n_iter):
			boot = np.random.choice(range(len(docs.columns)), 
									size=len(docs.columns), replace=True)
			for i, dom in enumerate(domains):
				dom_pmids = dom2docs[dom]
				dom_vecs = docs.loc[dom_pmids].values[:,boot]
				dom_arche = archetypes.values[boot,i].reshape(1, archetypes.shape[0])
				df_boot[i,n] = 1.0 - np.mean(cdist(dom_vecs, dom_arche, metric="dice"))
			if n % int(n_iter / 10.0) == 0:
				print("\t\tProcessed {} iterations".format(n))
		df_boot = pd.DataFrame(df_boot, index=domains, columns=range(n_iter))
		df_boot.to_csv(file_boot)
	else:
		df_boot = pd.read_csv(file_boot, index_col=0, header=0)

	stats = utilities.compare_to_null(df_null, df, domains, n_iter, alpha=alpha)

	archetype.plot_violins(framework, domains, stats, df_null, df_obs, utilities.palettes[framework], 
			 	 		   dx=gen_dx[framework], ds=gen_ds[framework], alphas=[0], interval=0.999, print_fig=False,
			 	 		   ylim=[-0.25,0.75], yticks=[-0.25,0,0.25,0.5,0.75], font=arial, path="archetype/")


### Article partitions ###
print("\nPlotting MDS of article partitions")

from mds import mds

partitions = {framework: pd.read_csv("partition/data/doc2dom_{}.csv".format(
									 framework), index_col=0, header=None) - 1.0 
			  for framework in frameworks}

colors = {}
for framework in frameworks:
	colors[framework] = [utilities.palettes[framework][int(partitions[framework].loc[pmid])] for pmid in pmids]

shapes = {"data-driven": ["o", "v", "^", ">", "<", "s"],
		  "rdoc": ["o", "v", "^", ">", "<", "s"],
		  "dsm": ["o", "v", "^", ">", "<", "s", "X", "D", "p"]}

markers = {}
for framework in frameworks:
	markers[framework] = [shapes[framework][int(partitions[framework].loc[pmid])] for pmid in pmids]

mds_metric = True
eps = 0.001
max_iter = 5000

mds_file = "mds/data/mds_metric{}_eps{}_iter{}.csv".format(int(mds_metric), eps, max_iter)
X = pd.read_csv(mds_file, index_col=0, header=0).values

for framework in frameworks:
	mds.plot_mds(X, framework, colors, markers, metric=mds_metric, eps=eps, max_iter=max_iter, 
			 	 path="mds/", print_fig=False)




