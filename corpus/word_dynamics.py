#!/usr/bin/python3

import os, re
import pandas as pd
import numpy as np


def flip_dic(dic):
	flipped = {val: [] for val in set(dic.values())}
	for key, val in dic.items():
		flipped[val].append(key)
	return flipped


def load_data_from_records(dir):

	import xml.etree.ElementTree as ET

	pmid2year = {}
	missing = []
	for file in os.listdir(dir):
		pmid = int(file.replace(".xml", ""))
		tree = ET.parse("{}/{}".format(dir, file))
		root = tree.getroot()
		try:
			year = root.find("MedlineCitation").find("Article").find("Journal").find("JournalIssue").find("PubDate").find("Year").text
			pmid2year[pmid] = int(year)
		except:
			try:
				date = root.find("MedlineCitation").find("Article").find("Journal").find("JournalIssue").find("PubDate").find("MedlineDate").text
				year = re.search(r"\b(19|20)\d{2}\b", date).group(0)
				pmid2year[pmid] = int(year)
			except:
				print("Missing PMID {}".format(pmid))
				missing.append(pmid)
	
	print("{} articles missing year info".format(len(missing)))
	
	year2pmids = flip_dic(pmid2year)
	return year2pmids


def load_data_from_meta(file):
	df = pd.read_csv(file, index_col=None, encoding="Latin-1")
	df = df.dropna(subset=["PMID"])
	df["PMID"] = df["PMID"].astype(int)
	df.index = df["PMID"]

	pmid2year = {pmid: row["YEAR"] for pmid, row in df.iterrows()}
	year2pmids = flip_dic(pmid2year)
	return year2pmids


def extract_freq_by_year(words, year2pmids, dir):

	years = sorted(list(set(year2pmids.keys())))
	freq = {word: {year: 0.0 for year in years} for word in words}
	ngrams = [word for word in words if " " in word]

	for year in years:
		for pmid in year2pmids[year]:
			infile = "{}/{}.txt".format(dir, int(pmid))
			text = open(infile, "r").read()
			for ngram in ngrams:
				text = text.replace(ngram, ngram.replace(" ", "_"))
			text = text.split()
			for word in words:
				if word.replace(" ", "_") in text:
					freq[word][year] += 1.0 / len(year2pmids[year])
	return freq


def boot_freq_by_year(words, year2pmids, dir, 
					  n_iter=5000, seed=42, verbose=False):
	
	np.random.seed(seed)

	freq_boot = {}
	for i in range(n_iter):
		if verbose:
			print("Bootstrap iteration {}".format(i))
		year2pmids_sample = {year: np.random.choice(list(pmids), 
							 size=len(pmids), replace=True) for year, pmids in year2pmids.items()}
		freq_boot[i] = extract_freq_by_year(words, year2pmids_sample, dir)
	return freq_boot


def plot_freq_over_years(freq, freq_boot, file):

	import matplotlib.pyplot as plt
	from matplotlib import font_manager
	from matplotlib import rcParams

	arial = "../style/Arial Unicode.ttf"
	prop_lg = font_manager.FontProperties(fname=arial, size=20)
	prop_xlg = font_manager.FontProperties(fname=arial, size=26)
	rcParams["axes.linewidth"] = 1.5

	years = range(1997, 2018)

	fig = plt.figure(figsize=(5, 4.5))
	ax = fig.add_axes([0,0,1,1])

	words = ["dsm", "rdoc", "machine learning"]
	labels = ["DSM", "RDoC", "Machine Learning"]
	lty = ["dotted", "dashed", "solid"]
	
	iters = sorted(list(freq_boot.keys()))
	boots = [{year: [] for year in years} for word in words]
	for i, word in enumerate(words):
		for n in iters:
			for year in years:
				boots[i][year].append(freq_boot[n][word][year])

	for i, word in enumerate(words):
		plt.plot(years, [freq[word][y] for y in years], label=labels[i],
				linestyle=lty[i], linewidth=2, color="black")
		lower_i = [sorted(boots[i][year])[int(len(iters)*.05)] for year in years]
		upper_i = [sorted(boots[i][year])[int(len(iters)*.95)] for year in years]
		plt.fill_between(years, y1=lower_i, y2=upper_i, 
						 color="gray", alpha=0.2)
		
	plt.xticks(fontproperties=prop_lg)
	plt.yticks(fontproperties=prop_lg)

	ax.xaxis.set_tick_params(width=2, length=5)
	ax.yaxis.set_tick_params(width=2, length=5)

	ax.set_xticks(range(2000, 2020, 5))
	plt.ylim([-0.003,0.066])

	plt.xlabel("Year", fontproperties=prop_xlg)
	plt.ylabel("Proportion of articles", fontproperties=prop_xlg)

	ax.xaxis.set_label_coords(0.5, -0.12)
	ax.yaxis.set_label_coords(-0.165, 0.5)

	ax.spines["right"].set_visible(False)
	ax.spines["top"].set_visible(False)

	ax.legend(loc="upper left", prop=prop_lg, edgecolor="none", 
			  handletextpad=0.5, borderpad=0.55, bbox_to_anchor=(0, 1.075))

	plt.savefig("figures/{}.png".format(file), dpi=250, bbox_inches="tight")


words = ["dsm", "rdoc", "machine learning"]
year2pmids = load_data_from_meta("metadata_filt_180811.csv")
dir = "../../../nlp/corpus"
freq = extract_freq_by_year(words, year2pmids, dir)
freq_boot = boot_freq_by_year(words, year2pmids, dir, n_iter=5000, verbose=True)
plot_freq_over_years(freq, freq_boot, "cog_words_by_year")


