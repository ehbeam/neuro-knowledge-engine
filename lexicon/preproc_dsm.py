#!/usr/bin/python

# Preprocesses tokens from the DSM-5 index


import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# Load WordNet lemmatizer from NLTK
lemmatizer = WordNetLemmatizer()

# Load English stop words from NLTK
stops = stopwords.words("english")


# Function for stemming, conversion to lowercase, and removal of punctuation
def preprocess(token):

	# Encode in UTF-8
	token = token.encode("utf-8")

	# Replace words containing slashes with two forms, one with spaces and one with nothing as replacement
	preproc = []
	if "/" in token:
		preproc.append(token.replace("/", " "))
		preproc.append(token.replace("/", ""))
		parts = token.split("/")
		word1 = parts[0].split()[-1]
		word2 = parts[1].split()[0]
		preproc.append(token.replace(word1, "").replace("/", ""))
		preproc.append(token.replace(word2, "").replace("/", ""))
	if " or " in token:
		preproc.append(token.replace(" or ", " "))
		parts = token.split(" or ")
		word1 = parts[0].split()[-1]
		word2 = parts[1].split()[0]
		preproc.append(token.replace(word1, ""))
		preproc.append(token.replace(word2, ""))
	if "-" in token:
		preproc.append(token.replace("-", " "))
		preproc.append(token.replace("-", ""))
	if ("(") in token:
		parts = token.replace(")", "(").split("(")
		preproc.append(parts[1].strip())
		del parts[1]
		preproc.append("".join(parts).replace("  ", " "))
		preproc.append(token.replace("(", "").replace(")", ""))
	preproc.append(token)

	for i, token in enumerate(preproc): 

		# Convert to lowercase and remove stop words, which contain punctuation
		filtered = []
		for word in token.lower().split():
			if word not in stops:
				word = "".join([char for char in word if char.isalpha() or char.isdigit() or char == " "])
				filtered.append(word)
		text = " ".join(filtered)

		# Perform lemmatization, excluding acronyms and names in lexicon or RDoC
		acronyms = ["abc", "aai", "adhd", "aids", "atq", "asam", "asi", "aqc", "asi", "asq", "ax", "axcpt", "axdpx", "bees", "bas", "bdm", "bis", "bisbas", "beq", "brief", "cai", "catbat", "cfq", "deq", "dlmo", "dospert", "dsm", "dsmiv", "dsm5", "ecr", "edi", "eeg", "eei", "ema", "eq", "fmri", "fne", "fss", "grapes", "hrv", "iri", "isi", "ius", "jnd", "leas", "leds", "locs", "poms", "meq", "mctq", "sans", "ippa", "pdd", "pebl", "pbi", "prp", "mspss", "nart", "nartr", "nih", "npu", "nrem", "pas", "panss", "qdf", "rbd", "rem", "rfq", "sam", "saps", "soc", "srs", "srm", "strain", "suds", "teps", "tas", "tesi", "tms", "ug", "upps", "uppsp", "vas", "wais", "wisc", "wiscr", "wrat", "wrat4", "ybocs", "ylsi"]
		names = ["american", "badre", "barratt", "battelle", "bartholomew", "becker", "berkeley", "conners", "corsi", "degroot", "dickman", "marschak", "beckerdegrootmarschak", "beery", "buktenica", "beerybuktenica", "benton", "bickel", "birkbeck", "birmingham", "braille", "brixton", "california", "cambridge", "cattell", "cattells", "chapman", "chapmans", "circadian", "duckworth", "duckworths", "eckblad", "edinburgh", "erickson", "eriksen", "eysenck", "fagerstrom", "fitts", "gioa", "glasgow", "golgi", "gray oral", "halstead", "reitan", "halsteadreitan", "hamilton", "hayling", "holt", "hooper", "hopkins", "horne", "ostberg", "horneostberg", "iowa", "ishihara", "kanizsa", "kaufman", "koechlin", "laury", "leiter", "lennox", "gastaut", "lennoxgastaut", "london", "macarthur", "maudsley", "mcgurk", "minnesota", "montreal", "morris", "mullen", "muller", "lyer", "mullerlyer", "munich", "parkinson", "pavlovian", "peabody", "penn", "penns", "piaget", "piagets", "pittsburgh", "porteus", "posner", "rey", "ostereith", "reyostereith", "reynell", "rivermead", "rutledge", "salthouse", "babcock", "spielberger", "spielbergers", "stanford", "binet", "shaver", "simon", "stanfordbinet", "sternberg", "stroop", "toronto", "trier", "yale", "brown", "umami", "uznadze", "vandenberg", "kuse", "vernier", "vineland", "warrington", "warringtons", "wason", "wechsler", "wisconsin", "yalebrown", "zimbardo", "zuckerman"]
		lemmas = []
		for word in text.split():
			if word in acronyms + names:
				lemmas.append(word)
			else:
				lemmas.append(lemmatizer.lemmatize(word))
		token = "_".join(lemmas)
		preproc[i] = token

	return preproc


# Initialize lexicon sets and lists
lexicon_set = set()


# Load tokens from the DSM-5 index
def load_txt(file, ngram_set):
	for line in open(file, "r").readlines():
		token = line.strip().lower()
		for t in preprocess(token):
			lexicon_set.add(t)

load_txt("dsm5_diagnoses.txt".format(type), lexicon_set)


# Export preprocessed ontology lexicon
lexicon = list(lexicon_set)
lexicon.sort()
lexicon_file = open("dsm5_diagnoses_preproc.txt", "w+")
for n in lexicon:
	if n != "":
		lexicon_file.write(n + "\n")
lexicon_file.close()

