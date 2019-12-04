"""
Calculate the coverage, diversity and compression score for each article and summary pair for the given dataset.
Author: Tianwei She
Date: Feb 2019
"""
# pylint: disable=C0103

import sys
import pickle as pkl
from fragments import Fragments

dataset_name = "multi_news"
src_fname = sys.argv[1]
tgt_fname = sys.argv[2]

coverage, density, compression = [], [], []
with open(src_fname, 'r') as src_file:
    with open(tgt_fname, 'r') as tgt_file:
        articles = src_file.readlines()
        summaries = tgt_file.readlines()
        N = len(articles)
        print("Number of examples: %d" % N)
        for i in range(N):
            if i % 1000 == 0:
                print(i)
            fragment = Fragments(summaries[i], articles[i])
            coverage.append(fragment.coverage())
            density.append(fragment.density())
            compression.append(fragment.compression())


pkl.dump(coverage, open('pkl_files/' + dataset_name + "_coverage.pk", "wb"))
pkl.dump(density, open('pkl_files/' + dataset_name + "_density.pk", "wb"))
pkl.dump(compression, open('pkl_files/' + dataset_name + "_compression.pk", "wb"))
