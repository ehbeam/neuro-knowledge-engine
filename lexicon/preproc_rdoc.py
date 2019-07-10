#!/usr/bin/python

# Preprocesses terms from behaviorally relevant cells of the RDoC matrix, including:
#	(1) Construct
#	(2) Description
#	(3) Behavior
#	(4) Self-Report
#	(5) Paradigms
# Filter 2-5 by the lexicon of combined ontologies


import string, os
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# Date of RDoC matrix download
date = 190124

# WordNet lemmatizer from NLTK
lemmatizer = WordNetLemmatizer()

# English stop words from NLTK combined with custom stop word list
stops = stopwords.words("english") 

# Symbols to split strings by
punc = string.punctuation

# Lexicon from combined ontologies
lexicon = [token.strip() for token in open("../../../nlp/lexicon/lexicon_cogneuro_preproc_man.txt").readlines()]


# Function to compute all n-grams in a string
def expand_ngrams(phrase):
	
	ngrams = []
	for n in range(2, min(6,len(phrase))):
		for i in range(len(phrase)-n+1):
			ngrams.append(phrase[i:i+n])
	
	return ngrams


def lemmatize(token):
	
	acronyms = ["abc", "aai", "adhd", "aids", "atq", "asam", "asi", "aqc", "asi", "asq", "ax", "axcpt", "axdpx", "bees", "bas", "bdm", "bis", "bisbas", "beq", "brief", "cai", "catbat", "cfq", "deq", "dlmo", "dospert", "dsm", "dsmiv", "dsm5", "ecr", "edi", "eeg", "eei", "ema", "eq", "fmri", "fne", "fss", "grapes", "hrv", "iri", "isi", "ius", "jnd", "leas", "leds", "locs", "poms", "meq", "mctq", "sans", "ippa", "pdd", "pebl", "pbi", "prp", "mspss", "nart", "nartr", "nih", "npu", "nrem", "pas", "panss", "qdf", "rbd", "rem", "rfq", "sam", "saps", "soc", "srs", "srm", "strain", "suds", "teps", "tas", "tesi", "tms", "ug", "upps", "uppsp", "vas", "wais", "wisc", "wiscr", "wrat", "wrat4", "ybocs", "ylsi"]
	names = ["american", "badre", "barratt", "battelle", "bartholomew", "becker", "berkeley", "conners", "corsi", "degroot", "dickman", "marschak", "beckerdegrootmarschak", "beery", "buktenica", "beerybuktenica", "benton", "bickel", "birkbeck", "birmingham", "braille", "brixton", "california", "cambridge", "cattell", "cattells", "chapman", "chapmans", "circadian", "duckworth", "duckworths", "eckblad", "edinburgh", "erickson", "eriksen", "eysenck", "fagerstrom", "fitts", "gioa", "glasgow", "golgi", "gray oral", "halstead", "reitan", "halsteadreitan", "hamilton", "hayling", "holt", "hooper", "hopkins", "horne", "ostberg", "horneostberg", "iowa", "ishihara", "kanizsa", "kaufman", "koechlin", "laury", "leiter", "lennox", "gastaut", "lennoxgastaut", "london", "macarthur", "maudsley", "mcgurk", "minnesota", "montreal", "morris", "mullen", "muller", "lyer", "mullerlyer", "munich", "parkinson", "pavlovian", "peabody", "penn", "penns", "piaget", "piagets", "pittsburgh", "porteus", "posner", "rey", "ostereith", "reyostereith", "reynell", "rivermead", "rutledge", "salthouse", "babcock", "spielberger", "spielbergers", "stanford", "binet", "shaver", "simon", "stanfordbinet", "sternberg", "stroop", "toronto", "trier", "yale", "brown", "umami", "uznadze", "vandenberg", "kuse", "vernier", "vineland", "warrington", "warringtons", "wason", "wechsler", "wisconsin", "yalebrown", "zimbardo", "zuckerman"]
	if token not in acronyms + names:
		return lemmatizer.lemmatize(token)
	else:
		return token


# Function to preprocess a single word
def preprocess(word):
	
	word = "".join(char for char in word.lower() if char not in punc)
	word = lemmatize(word)
	
	return word


# Function for preprocessing a construct name
def preprocess_construct(construct):
	
	if "/" in construct:
		parts = construct.split("/")
		word1 = parts[0].split()[-1]
		word2 = parts[1].split()[0]
		v1 = construct.replace(word1, "").replace("/", "")
		v2 = construct.replace(word2, "").replace("/", "")
		construct = " | ".join([v1, v2])
	
	if " and " in construct:
		parts = construct.split(" and ")
		construct = construct.replace(" and ", " ")
		part1 = construct.replace(parts[0].split()[-1], "").strip()
		part2 = construct.replace(parts[1].split()[0], "").strip()
		construct = " | ".join([part1, part2])
	
	tokens = construct.replace(", ", " | ").split(" | ")
	tokens = ["_".join([preprocess(word) for word in token.split() if word not in stops]) for token in tokens]
	
	return tokens


# Function to expand token into versions with and without custom stop words
def expand_stops(word):
	
	custom_stops = ["checklist", "interview", "inventory", "paradigm", "questionnaire", "scale", "schedule", "subscale", "task", "test", "testing"]
	
	words = [word]
	for token in word.split():
		for stop in custom_stops:
			if token == stop:
				words += [word.replace(stop, "").strip().replace("  ", " ")]
	
	words = [word for word in words if word not in ["", " "]]
	
	return words


# Function for preprocessing a single token
def preprocess_word2tkn(word):
	
	word = " ".join([preprocess(tkn) for tkn in word.split() if tkn not in stops])
	tokens = [token.replace(" ", "_") for token in expand_stops(word)]
	
	return tokens


# Function for preprocessing multiword cells
def preprocess_cell2lex(cell):
	
	tokens = [tkn.replace("/", " ") for tkn in cell.split()]
	tokens = [" ".join([preprocess(tkn) for tkn in ngram if tkn not in stops]) for ngram in expand_ngrams(tokens)]
	tokens = ["_".join([token for token in expand_stops(word)]) for word in tokens]
	tokens = [token.replace(" ", "_") for token in tokens if token.replace(" ", "_") in lexicon]
	
	return tokens


# Load RDoC matrix and units to preprocess
rdoc = pd.read_csv("../data/rdoc_{}/rdoc_man_{}.csv".format(date, date))
units = ["CONSTRUCT", "DESCRIPTION", "BEHAVIOR", "SELF_REPORT", "PARADIGMS"]

# Initialize dictionary and set for storing seed terms
matrix = {con: {} for con in rdoc["CONSTRUCT_NAME"]}
seeds, seed_df, j = [], {}, 0
cons = {con: [] for con in rdoc["CONSTRUCT_NAME"]}
desc = {}

# Load and preprocess matrix
for unit, col in rdoc[units].iteritems():
	for i, row in col.iteritems():
		con = rdoc["CONSTRUCT_NAME"][i]
		dom = rdoc["DOMAIN_NAME"][i]
		matrix[con]["DOMAIN_NAME"] = dom
		matrix[con]["CONSTRUCT_NAME"] = con
		tokens = []
		if not isinstance(row, float):
			for cell in row.split(" | "):
				if unit == "CONSTRUCT":
					tokens += preprocess_construct(cell)
				else:
					if cell.count(" ") < 6:
						tokens += preprocess_word2tkn(cell)
					tokens += preprocess_cell2lex(cell)
		tokens = sorted(list(set(tokens)))
		if unit == "DESCRIPTION":
			for token in tokens:
				if token not in desc:
					desc[token] = 0
				desc[token] += 1

		matrix[con][unit] = ", ".join(tokens)
		seeds += tokens
		for token in tokens:
			if token not in cons[con]:
				seed_df[j] = {}
				seed_df[j]["TOKEN"] = token
				seed_df[j]["DOMAIN"] = dom
				seed_df[j]["CONSTRUCT"] = con
				seed_df[j]["UNIT"] = unit
				j += 1
		cons[con] += tokens

# Export preprocessed RDoC matrix
df = pd.DataFrame(matrix, index=["DOMAIN_NAME", "CONSTRUCT_NAME"]+units, columns=rdoc["CONSTRUCT_NAME"]).transpose()
df.to_csv("../data/rdoc_{}/rdoc_preproc.csv".format(date), index=None)

# Export list of raw terms
seeds = sorted(list(set(seeds)))
with open("../data/rdoc_{}/rdoc_seeds.txt".format(date), "w+") as seed_file:
	seed_file.write("\n".join(seeds))

# Count seeds by construct
counts = {seed: 0 for seed in seeds}
for con, units in df.iterrows():
	for unit in units:
		for token in unit.split(", "):
			if token in seeds:
				counts[token] += 1

# Reorder data frame of seeds
con_i = {con: i for i, con in enumerate(rdoc["CONSTRUCT_NAME"])}
cols = ["DOMAIN", "CONSTRUCT", "UNIT", "TOKEN", "CONFIDENCE"]
seed_df = pd.DataFrame(seed_df, index=cols).transpose()
seed_df["ORDER"] = [con_i[con] for con in seed_df["CONSTRUCT"]]

# Compute confidence of construct tokens
# Confidence = 1 if a token is unique to a construct
seed_df["CONFIDENCE"] = 0
con_counts = {token: len(seed_df.loc[(seed_df["UNIT"] == "CONSTRUCT") & (seed_df["TOKEN"] == token)]) for token in set(list(seed_df.loc[seed_df["UNIT"] == "CONSTRUCT", "TOKEN"]))}
unique_cons = [con for con, count in con_counts.items() if count == 1]
seed_df.loc[(seed_df["UNIT"] == "CONSTRUCT") & (seed_df["TOKEN"].isin(unique_cons)), "CONFIDENCE"] = 1
# seed_df.loc[(seed_df["UNIT"] == "CONSTRUCT"), "CONFIDENCE"] = 1

# Remove description tokens appearing in more than one construct
redundant_desc = [token for token, count in desc.items() if count > 1]
seed_df = seed_df[~((seed_df["UNIT"] == "DESCRIPTION") & (seed_df["TOKEN"].isin(redundant_desc)))]

# Export data frame of seeds
seed_df = seed_df.sort_values(["ORDER", "CONFIDENCE", "UNIT", "TOKEN"], ascending=[True,False,True,True])
seed_df.to_csv("../data/rdoc_{}/rdoc_seeds.csv".format(date), index=None, columns=["ORDER"]+cols)

