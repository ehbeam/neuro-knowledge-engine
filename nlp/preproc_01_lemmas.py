#!/usr/bin/python

# Preprocess raw full texts in the following steps:
#   (1) Case-folding and punctuation removal
#   (2) Lemmatization with WordNet
#   (3) Consolidation of n-grams from RDoC and select mental functioning ontology

import os, re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd

# Date of RDoC data download
date = 180716

# Load WordNet lemmatizer from NLTK
lemmatizer = WordNetLemmatizer()

# Load English stop words from NLTK
stops = stopwords.words("english")

# Load PMIDs for CogNeuro and PubMed datasets
pmids_cogneuro = [str(int(pmid)) for pmid in pd.read_csv("../cogneuro/metadata/metadata_raw_180811.csv", index_col=None)["PMID"]]
pmids_pubmed = [pmid.strip() for pmid in open("../pubmed/query_180731/pmids.txt").readlines()]

def lemmatize(token):
	acronyms = ["abc", "aai", "adhd", "aids", "atq", "asam", "asi", "aqc", "asi", "asq", "ax", "axcpt", "axdpx", "bees", "bas", "bdm", "bis", "bisbas", "beq", "brief", "cai", "catbat", "cfq", "deq", "dlmo", "dospert", "dsm", "dsmiv", "dsm5", "ecr", "edi", "eeg", "eei", "ema", "eq", "fmri", "fne", "fss", "grapes", "hrv", "iri", "isi", "ius", "jnd", "leas", "leds", "locs", "poms", "meq", "mctq", "sans", "ippa", "pdd", "pebl", "pbi", "prp", "mspss", "nart", "nartr", "nih", "npu", "nrem", "pas", "panss", "qdf", "rbd", "rem", "rfq", "sam", "saps", "soc", "srs", "srm", "strain", "suds", "teps", "tas", "tesi", "tms", "ug", "upps", "uppsp", "vas", "wais", "wisc", "wiscr", "wrat", "wrat4", "ybocs", "ylsi"]
	names = ["american", "badre", "barratt", "battelle", "bartholomew", "becker", "berkeley", "conners", "corsi", "degroot", "dickman", "marschak", "beckerdegrootmarschak", "beery", "buktenica", "beerybuktenica", "benton", "bickel", "birkbeck", "birmingham", "braille", "brixton", "california", "cambridge", "cattell", "cattells", "chapman", "chapmans", "circadian", "duckworth", "duckworths", "eckblad", "edinburgh", "erickson", "eriksen", "eysenck", "fagerstrom", "fitts", "gioa", "glasgow", "golgi", "gray oral", "halstead", "reitan", "halsteadreitan", "hamilton", "hayling", "holt", "hooper", "hopkins", "horne", "ostberg", "horneostberg", "iowa", "ishihara", "kanizsa", "kaufman", "koechlin", "laury", "leiter", "lennox", "gastaut", "lennoxgastaut", "london", "macarthur", "maudsley", "mcgurk", "minnesota", "montreal", "morris", "mullen", "muller", "lyer", "mullerlyer", "munich", "parkinson", "pavlovian", "peabody", "penn", "penns", "piaget", "piagets", "pittsburgh", "porteus", "posner", "rey", "ostereith", "reyostereith", "reynell", "rivermead", "rutledge", "salthouse", "babcock", "spielberger", "spielbergers", "stanford", "binet", "shaver", "simon", "stanfordbinet", "sternberg", "stroop", "toronto", "trier", "yale", "brown", "umami", "uznadze", "vandenberg", "kuse", "vernier", "vineland", "warrington", "warringtons", "wason", "wechsler", "wisconsin", "yalebrown", "zimbardo", "zuckerman"]
	if token not in acronyms + names:
		return lemmatizer.lemmatize(token)
	else:
		return token

# Function for stemming, conversion to lowercase, and removal of punctuation
def preprocess(text):

	# Convert to lowercase, convert slashes to spaces, and remove remaining punctuation except periods
	text = text.replace("-\n", "").replace("\n", " ").replace("\t", " ")
	text = "".join([char for char in text.lower() if char.isalpha() or char.isdigit() or char in [" ", "."]])
	text = text.replace(".", " . ").replace("  ", " ").strip()
	text = re.sub("\. \.+", ".", text)

	# Perform lemmatization, excluding acronyms and names in RDoC matrix
	text = " ".join([lemmatize(token) for token in text.split() if token not in stops])
	return text

def run_open_access(pmid, infile):
	if os.path.isfile(infile):
		outfile = "corpus/{}.txt".format(pmid)
		text = preprocess(open(infile, "rb").read())
		with open(outfile, "w+") as fout:
			fout.write(text)

def run_extracted(pmid, infile):
	if os.path.isfile(infile):
		outfile = "corpus/{}.txt".format(pmid)
		if not os.path.isfile(outfile):
			text = preprocess(open(infile, "rb").read())
			with open(outfile, "w+") as fout:
				fout.write(text)

# Instantiate outfile for preprocessed free text
if not os.path.exists("corpus".format(date)):
	os.makedirs("corpus".format(date))

# Loop over PMIDs in CogNeuro then PubMed, overwriting with Open Access full texts
for pmid in pmids_cogneuro:
	infile = "../cogneuro/texts/open_access/raw/{}.txt".format(pmid)
	run_open_access(pmid, infile)
for pmid in pmids_pubmed:
	infile = "../pubmed/open_access/raw/{}.txt".format(pmid)
	run_open_access(pmid, infile)

# Loop over PMIDs in CogNeuro then PubMed, adding in texts extracted from PDFs
for pmid in pmids_cogneuro:
	infile = "../cogneuro/texts/raw/{}.txt".format(pmid)
	run_extracted(pmid, infile)
for pmid in pmids_pubmed:
	infile = "../pubmed/raw/{}.txt".format(pmid)
	run_extracted(pmid, infile)

# Loop over PMIDs in PubMed, adding in texts extracted from ACE html
for pmid in pmids_pubmed:
	infile = "../pubmed/ace/raw/{}.txt".format(pmid)
	run_extracted(pmid, infile)

# Check that PMIDs removed from the corpora haven't snuck in
for pmid in [file.replace(".txt", "") for file in os.listdir("corpus") if not file.startswith(".")]:
	if pmid not in pmids_cogneuro + pmids_pubmed:
		print("Removed from ID list: ".format(pmid))

