# Created by Elizabeth Beam on 7/14/17
# Concatenates text files into one output file
# Example function call: sew("/Users/ehbeam/Dropbox/Stanford/Research/Projects/Psychiatlas/program", "/texts/preproc", "/data/fulltext_preproc.txt")

import os, re

def sew(dir, outfile):

    # Reset outfile
    open(outfile, "w+").close()
    fout = open(outfile, "a")

    # Loop over indir files and write to outfile
    for pmid in [file.replace(".txt", "") for file in os.listdir(dir) if not file.startswith(".")]:
        text = open("{}/{}.txt".format(dir, str(pmid)), "r").read()
        text = re.sub("\. \.+", ".", text).strip()
        fout.write(text + "\n\n")
    fout.close()

date = 190124
sew("corpus", "corpus_{}.txt".format(date))