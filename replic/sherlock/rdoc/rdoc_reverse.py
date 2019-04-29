#!/usr/bin/python

import os, pickle, random
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import binarize
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from time import time

def load_tokens(file, filter=[]):
	tokens = [token.strip() for token in open(file).readlines()]
	if len(filter) > 0:
		tokens = [token for token in tokens if token in filter]
	return sorted(tokens)

def load_split(split):
	file = "../../data/splits/{}.txt".format(split)
	return [int(id.strip()) for id in open(file).readlines()]

def mean_thres(df):
	col_mean = df.mean()
	df_bin = np.empty((df.shape[0], df.shape[1]))
	i = 0
	for col, doc_mean in col_mean.iteritems():
		df_bin[:,i] = 1 * (df[col] > doc_mean)
		i += 1
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin

def score_lists(lists, dtm, label_var="LABEL"):
	dtm = pd.DataFrame(binarize(dtm, threshold=0), index=dtm.index, columns=dtm.columns)
	labels = OrderedDict.fromkeys(lists[label_var])
	list_counts = pd.DataFrame(index=dtm.index, columns=labels)
	for label in list_counts.columns:
		tkns = lists.loc[lists[label_var] == label, "TOKEN"]
		tkns = [token for token in tkns if token in dtm.columns]
		list_counts[label] = dtm[tkns].sum(axis=1)
	list_scores = mean_thres(list_counts)
	return list_scores

def report(results, n_top=3):
	for i in range(1, n_top + 1):
		candidates = np.flatnonzero(results["rank_test_score"] == i)
		for candidate in candidates:
			print("-" * 30 + "\nModel rank: {0}".format(i))
			print("Cross-validated score: {0:.4f} (std: {1:.4f})".format(
				  results["mean_test_score"][candidate],
				  results["std_test_score"][candidate]))
			print("Parameters: {0}".format(results["params"][candidate]))

def search_grid(X, Y, param_grid, clf, scoring="roc_auc"):
	grid_search = GridSearchCV(clf, param_grid=param_grid, scoring=scoring, cv=5, n_jobs=5)
	start = time()
	grid_search.fit(X, Y)
	print("\nGridSearchCV took %.2f seconds for %d candidate parameter settings\n"
		  % (time() - start, len(grid_search.cv_results_["params"])))
	report(grid_search.cv_results_)
	return grid_search

def evaluate(X, Y, classifier, validation):
	classes = Y.columns
	columns = ["TP", "FP", "TN", "FN", "TPR", "TNR", "PPV", "NPV", "F1", "ACCURACY", "ROC_AUC", "LOG_LOSS"]
	df = pd.DataFrame(columns=columns)
	predictions = classifier.predict(X.loc[validation])
	probabilities = classifier.predict_proba(X.loc[validation])
	for j in range(predictions.shape[1]):
		true_labels = Y.loc[validation, classes[j]]
		pred_labels = predictions[:,j]
		tp = float(np.sum(np.logical_and(pred_labels == 1, true_labels == 1)))
		tn = float(np.sum(np.logical_and(pred_labels == 0, true_labels == 0)))
		fp = float(np.sum(np.logical_and(pred_labels == 1, true_labels == 0)))
		fn = float(np.sum(np.logical_and(pred_labels == 0, true_labels == 1)))
		tpr, tnr, ppv, npv, f1, acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		if (tp + fn) > 0:
			tpr = tp / (tp + fn)
		if (tn + fp) > 0:
			tnr = tn / (tn + fp)
		if (tp + fp) > 0:
			ppv = tp / (tp + fp)
		if (tn + fn) > 0:
			npv = tn / (tn + fn)
		if ((2 * tp) + fp + fn) > 0:
			f1 = (2 * tp) / ((2 * tp) + fp + fn)
		if (tp + tn + fp + fn) > 0:
			acc = (tp + tn) / (tp + tn + fp + fn)
		try:
			ll = log_loss(true_labels, pred_labels)
		except:
			ll = np.nan
			print("Failed to compute log loss for {}".format(classes[j]))
		try:
			auc = roc_auc_score(true_labels, probabilities[:,j])
		except:
			auc = np.nan
			print("Failed to compute ROC-AUC for {}".format(classes[j]))
		df = df.append(pd.DataFrame({"TP": [tp], "FP": [fp], "TN": [tn], "FN": [fn], "TPR": [tpr], "TNR": [tnr], "PPV": [ppv], "NPV": [npv], "F1": [f1], "ACCURACY": [acc], "ROC_AUC": [auc], "LOG_LOSS": [ll]}, index=[classes[j]], columns=columns))
	return df
	
def validate_classifier(X, Y, classifier, train, validation):
	classifier.fit(X.loc[train], Y.loc[train])
	df = evaluate(X, Y, classifier, validation)
	return classifier, df

def validate_null_classifier(X, Y, Y_null, classifier, train, validation):
	classifier.fit(X.loc[train], Y_null.loc[train])
	df = evaluate(X, Y, classifier, validation)
	return classifier, df

def resample_classifier(level, features, X, Y, classifier, train, validation, seed=42, list_len=25, n_samples=100, verbose=True):
	
	fit, df = validate_classifier(X, Y, classifier, train, validation)
	columns = ["N_COMPONENTS", "MAX_LIST_LEN"] + list(df.columns)
	if level == "domain":
		n_components = 5
	if level == "construct":
		n_components = 42
	df["N_COMPONENTS"] = n_components
	if features == "lists":
		df["MAX_LIST_LEN"] = list_len
	df["LABEL"] = df.index
	df.to_csv("results/reverse_rdoc_{}_{}.csv".format(level, features), 
			  index=None, columns=["LABEL"] + columns)

	boot_df = pd.DataFrame()
	i = 0
	for n in range(n_samples):
		random.seed(n+seed)
		sample = [random.choice(train) for t in train]
		n_fit, n_df = validate_classifier(X, Y, classifier, sample, validation)
		n_df["LABEL"] = df.index
		boot_df = boot_df.append(n_df)
		i += 1
		if i == 50:
			print("Assessed {:02d} resampling iterations".format(n+1))
			boot_df["N_COMPONENTS"] = n_components
			if features == "lists":
				boot_df["MAX_LIST_LEN"] = list_len
			boot_df.to_csv("results/reverse_rdoc_{}_{}_boot.csv".format(level, features), 
						   index=None, columns=["LABEL"] + columns)
			i = 0
	boot_df["N_COMPONENTS"] = n_components
	if features == "lists":
		boot_df["MAX_LIST_LEN"] = list_len
	boot_df.to_csv("results/reverse_rdoc_{}_{}_boot.csv".format(level, features), 
				   index=None, columns=["LABEL"] + columns)
	if verbose:
		print("\n      Model    Boot")
		print("PPV:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["PPV"].mean()*100, boot_df["PPV"].mean()*100, boot_df["PPV"].std()*100))
		print("TPR:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["TPR"].mean()*100, boot_df["TPR"].mean()*100, boot_df["TPR"].std()*100))
		print("TNR:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["TNR"].mean()*100, boot_df["TNR"].mean()*100, boot_df["TNR"].std()*100))
		print("F1:   {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["F1"].mean()*100, boot_df["F1"].mean()*100, boot_df["F1"].std()*100))
		print("ACC:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["ACCURACY"].mean()*100, boot_df["ACCURACY"].mean()*100, boot_df["ACCURACY"].std()*100))
	results = {"FIT": fit, "DF": df, "BOOT_DF": boot_df}
	return results

def permute_classifier(level, features, X, Y, classifier, train, validation, seed=42, list_len=25, n_shuffles=100, verbose=True):
	
	fit, df = validate_classifier(X, Y, classifier, train, validation)
	columns = ["N_COMPONENTS", "MAX_LIST_LEN"] + list(df.columns)
	if level == "domain":
		n_components = 5
	if level == "construct":
		n_components = 42
	df["N_COMPONENTS"] = n_components
	if features == "lists":
		df["MAX_LIST_LEN"] = list_len
	df["LABEL"] = df.index
	df.to_csv("results/reverse_rdoc_{}_{}.csv".format(level, features), 
			  index=None, columns=["LABEL"]+columns)

	null_fit, top_f1 = classifier, 0
	null_df = pd.DataFrame()
	i = 0
	for n in range(n_shuffles):
		random.seed(n+seed)
		Y_shuffled = np.array([random.sample(list(Y.iloc[:,j].values), Y.shape[0]) for j in range(Y.shape[1])])
		Y_shuffled = pd.DataFrame(Y_shuffled.transpose(), index=Y.index, columns=Y.columns)
		n_fit, n_df = validate_null_classifier(X, Y, Y_shuffled, classifier, train, validation)
		n_df["LABEL"] = df.index
		if n_df["F1"].mean() > top_f1:
			null_fit = n_fit
		null_df = null_df.append(n_df)
		i += 1
		if i == 50:
			print("Assessed {:02d} shuffle iterations".format(n+1))
			null_df["N_COMPONENTS"] = n_components
			if features == "lists":
				null_df["MAX_LIST_LEN"] = list_len
			null_df.to_csv("results/reverse_rdoc_{}_{}_null.csv".format(level, features), 
						   index=None, columns=["LABEL"] + columns)
			i = 0
	null_df["N_COMPONENTS"] = n_components
	if features == "lists":
		null_df["MAX_LIST_LEN"] = list_len
	null_df.to_csv("results/reverse_rdoc_{}_{}_null.csv".format(level, features), 
				   index=None, columns=["LABEL"] + columns)
	if verbose:
		print("\n      Model    Null")
		print("PPV:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["PPV"].mean()*100, null_df["PPV"].mean()*100, null_df["PPV"].std()*100))
		print("TPR:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["TPR"].mean()*100, null_df["TPR"].mean()*100, null_df["TPR"].std()*100))
		print("TNR:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["TNR"].mean()*100, null_df["TNR"].mean()*100, null_df["TNR"].std()*100))
		print("F1:   {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["F1"].mean()*100, null_df["F1"].mean()*100, null_df["F1"].std()*100))
		print("ACC:  {:06.2f}%  {:06.2f}% (+/- {:06.2f}% SD)".format(
		  df["ACCURACY"].mean()*100, null_df["ACCURACY"].mean()*100, null_df["ACCURACY"].std()*100))
	results = {"FIT": fit, "DF": df, "NULL_FIT": null_fit, "NULL_DF": null_df}
	return results

def run(level, features, list_len=25, dtm_thres=0, optimize_clf=False,
		penalty=["l1", "l2"], C=[0.001, 0.01, 0.1, 1, 10, 100, 1000], intercept=[True, False], 
		bootstrap=True, n_samples=1000, permutation=True, n_shuffles=1000, seed=1):

	train = load_split("train")
	validation = load_split("validation")
	test = load_split("test")
	validation = validation + test
	
	activations = pd.read_csv("../data/dcm_0mm_thres0.csv", index_col=0)

	dtm = pd.read_csv("../data/dtm_190124.csv.gz", compression="gzip", index_col=0)
	dtm = mean_thres(dtm)

	lists = pd.read_csv("../data/lists_rdoc_{}_opsim.csv".format(level), index_col=None)
	scores = score_lists(lists, dtm, label_var=level.upper())

	print("-" * 50 + "\nRDoC {}, {}\n".format(level, features) + "-" * 50)
	
	validation = [pmid for pmid in validation if (pmid in scores.index) and (pmid in activations.index)]
	
	fit_file = "results/reverse_rdoc_{}_opsim.p".format(level)
	if optimize_clf == True or not os.path.isfile(fit_file):
		
		param_grid = {"estimator__penalty": penalty,
				  	  "estimator__C": C,
					  "estimator__fit_intercept": intercept,
					  "estimator__solver": ["liblinear"],
					  "estimator__random_state": [42], 
					  "estimator__max_iter": [1000], 
					  "estimator__tol": [1e-10]}

		classifier = OneVsRestClassifier(LogisticRegression())

		cv_results = search_grid(activations.loc[train], scores.loc[train], param_grid, classifier, scoring="roc_auc")
		top_clf = cv_results.best_estimator_
		pickle.dump(top_clf, open(fit_file, "wb"), protocol=2)
	
	else:
		top_clf = pickle.load(open(fit_file, "rb"))
	
	if bootstrap:
		print("")
		boot = resample_classifier(level, features, activations, scores, top_clf, 
						   		   seed=seed, train=train, validation=validation, list_len=list_len, n_samples=n_samples)

	if permutation:
		print("")
		null = permute_classifier(level, features, activations, scores, top_clf, 
							      seed=seed, train=train, validation=validation, list_len=list_len, n_shuffles=n_shuffles)

# run("domain", "lists", optimize_clf=True, bootstrap=False, permutation=False)
