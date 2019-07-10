#!/usr/bin/python

# Consolidates n-grams from RDoC and select mental functioning ontologies


import os, re
import pandas as pd


# Date of RDoC data download
date = 190124

# Load n-grams from RDoC and select ontologies
ngrams_rdoc = [word.strip().replace("_", " ") for word in open("../rdoc/rdoc-preproc/data/rdoc_{}/rdoc_seeds.txt".format(date), "r").readlines() if "_" in word]
ngrams_cogneuro = [word.strip().replace("_", " ") for word in open("lexicon/lexicon_cogneuro_preproc.txt", "r").readlines() if "_" in word]
ngrams_dsm = [word.strip().replace("_", " ") for word in open("lexicon/dsm5_seeds.txt", "r").readlines() if "_" in word]
ngrams_psych = [word.strip().replace("_", " ") for word in open("lexicon/lexicon_psychiatry_preproc.txt", "r").readlines() if "_" in word]
ngrams = list(set(ngrams_rdoc + ngrams_cogneuro + ngrams_dsm + ngrams_psych))
ngrams.sort(key = lambda x: x.count(" "), reverse = True)


# Function to consolidate n-grams from RDoC then select ontologies
def preprocess(text):

	text = text.replace("_", " ")
	for ngram in ngrams:
		text = text.replace(ngram, ngram.replace(" ", "_"))
	text = re.sub("\. \.+", ".", text)

	return text


# Loop over files in corpus directory
pmids = [file.replace(".txt", "") for file in os.listdir("corpus") if not file.startswith(".")]
for pmid in pmids:
	file = "corpus/{}.txt".format(pmid)
	text = preprocess(open(file, "rb").read())
	with open(file, "w+") as fout:
		fout.write(text)
