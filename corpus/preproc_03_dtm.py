#!/usr/bin/python

# Compute DTM for the cognitive neuroscience lexicon

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Date RDoC data was downloaded
date = 190124

# Load and combine lexicons
lex_rdoc = [word.strip().replace(" ", "_") for word in open("../data/rdoc_{}/rdoc_seeds.txt".format(date), "r").readlines()]
lex_cogneuro = [word.strip().replace(" ", "_") for word in open("../data/lexicon_cogneuro_preproc.txt", "r").readlines()]
lexicon = sorted(list(set(lex_rdoc + lex_cogneuro)))

# Load preprocessed cogneuro texts
df = pd.read_csv("../data/metadata_filt_180811.csv", index_col=None, header=0)
df = df[df["PMID"] > 0]
pmids = [str(int(pmid)) for pmid in list(df["PMID"])]
records = [open("../../../nlp/corpus/{}.txt".format(pmid), "r").read() for pmid in pmids]

# Compute DTM with restricted vocabulary
vec = CountVectorizer(min_df=0, vocabulary=lexicon)
print("Fitting DTM of seeds and candidate synonyms...")
dtm = vec.fit_transform(records)
dtm_df = pd.DataFrame(dtm.toarray(), index=pmids, columns=lexicon)
dtm_df.to_csv("../data/dtm_{}.csv.gz".format(date), compression="gzip", index=pmids, columns=lexicon)