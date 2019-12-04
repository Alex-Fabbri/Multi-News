# pylint: disable=C0103
import os
from collections import Counter

import nltk
from nltk.tokenize import ToktokTokenizer
from nltk import sent_tokenize
import numpy as np
import dill

# TODO count the number of words per summary/ summary stats:
toktok = ToktokTokenizer()
num_sents_list = []
num_words_list = []
word_counter = Counter()
word_counter_lower = Counter()
for count, file in enumerate(os.listdir("/data/corpora/newser/data-final")):
    print(count)
    file_data = open(f"/data/corpora/newser/data-final/{file}/{file}.reference.txt", "r").read()
    summary_sents = sent_tokenize(file_data)
    num_sents_list.append(len(summary_sents))
    num_words = 0
    for sent in summary_sents:
        cur_words = toktok.tokenize(sent)
        word_counter.update(cur_words)
        word_counter_lower.update([word.lower() for word in cur_words])
        num_words += len(cur_words)
    num_words_list.append(num_words)
num_sents_np = np.array(num_sents_list)
num_words_np = np.array(num_words_list)

print(f"the average number of sentences per summary: {np.mean(num_sents_np)}\n")
print(f"the std of sentences per summary: {np.std(num_sents_np)}\n")
print(f"the average number of words per summary: {np.mean(num_words_np)}\n")
print(f"the std of words per summary: {np.std(num_words_np)}\n")


print(len(word_counter))
print(len(word_counter_lower))

# TODO calculate statistics for source documents
num_sents_list_indiv = []
num_words_list_indiv = []
num_sents_list_cat = []
num_words_list_cat = []
word_counter = Counter()
word_counter_lower = Counter()
id2files = dill.load(open("id2files.dill", "rb"))
for count, file in enumerate(os.listdir("/data/corpora/newser/data-final")):
    print(count)
    num_words_cur_id = 0
    num_sents_cur_id = 0
    for filename in id2files[file]:
        file_data = open(f"/data/corpora/newser/data-final/{file}/{filename}", "r").read()
        summary_sents = sent_tokenize(file_data)
        cur_num_sents = len(summary_sents)
        num_sents_list_indiv.append(cur_num_sents)
        num_sents_cur_id += cur_num_sents
        num_words_cur_file = 0
        for sent in summary_sents:
            cur_words = toktok.tokenize(sent)
            num_words_cur_file += len(cur_words)
            word_counter.update(cur_words)
            word_counter_lower.update([word.lower() for word in cur_words])
        num_words_list_indiv.append(num_words_cur_file)
        num_words_cur_id += num_words_cur_file
    num_sents_list_cat.append(num_sents_cur_id)
    num_words_list_cat.append(num_words_cur_id)
num_sents_np = np.array(num_sents_list_indiv)
num_words_np = np.array(num_words_list_indiv)

print(f"the average number of sentences per source document: {np.mean(num_sents_np)}\n")
print(f"the std of sentences per source document: {np.std(num_sents_np)}\n")
print(f"the average number of words per source document: {np.mean(num_words_np)}\n")
print(f"the std of words per source document: {np.std(num_words_np)}\n")
print("==============================================================================")

num_sents_np = np.array(num_sents_list_cat)
num_words_np = np.array(num_words_list_cat)
print(f"the average number of sentences per source documents concatenated: {np.mean(num_sents_np)}\n")
print(f"the std of sentences per source documents concatenated: {np.std(num_sents_np)}\n")
print(f"the average number of words per source documents concatenated: {np.mean(num_words_np)}\n")
print(f"the std of words per source documents concatenated: {np.std(num_words_np)}\n")
print("==============================================================================")

print(len(word_counter))
print(len(word_counter_lower))


#TODO calculate percent of novel n grams:
percent_new_unigrams = []
percent_new_bigrams = []
percent_new_trigrams = []
percent_new_4grams = []

for count, file in enumerate(os.listdir("/data/corpora/newser/data-final")):
    print(count)
    cur_word_list_source = []
    for filename in id2files[file]:
        file_data = open(f"/data/corpora/newser/data-final/{file}/{filename}", "r").read()
        source_sents = sent_tokenize(file_data.lower())

        for sent in source_sents:
            cur_words = toktok.tokenize(sent)
            cur_word_list_source.extend(cur_words)

    # my_bigrams_source = list(nltk.bigrams(cur_word_list_source))
    my_bigrams_source = [f"{tup[0]} {tup[1]}" for tup in nltk.bigrams(cur_word_list_source)]
    my_trigrams_source = [f"{tup[0]} {tup[1]} {tup[2]}" for tup in nltk.trigrams(cur_word_list_source)]
    my_4grams_source = [f"{tup[0]} {tup[1]} {tup[2]} {tup[3]}" for tup in nltk.ngrams(cur_word_list_source, 4)]

    cur_word_list_summ = []
    reference_data = open(f"/data/corpora/newser/data-final/{file}/{file}.reference.txt", "r").read()
    summary_sents = sent_tokenize(reference_data.lower())
    for sent in summary_sents:
        cur_words = toktok.tokenize(sent)
        cur_word_list_summ.extend(cur_words)

    my_bigrams_summ = [f"{tup[0]} {tup[1]}" for tup in nltk.bigrams(cur_word_list_summ)]
    my_trigrams_summ = [f"{tup[0]} {tup[1]} {tup[2]}" for tup in nltk.trigrams(cur_word_list_summ)]
    my_4grams_summ = [f"{tup[0]} {tup[1]} {tup[2]} {tup[3]}" for tup in nltk.ngrams(cur_word_list_summ, 4)]

    diff_unigram = np.setdiff1d(cur_word_list_summ, cur_word_list_source)
    percent_new_unigrams.append(len(diff_unigram)/len(cur_word_list_summ))

    diff_bigram = np.setdiff1d(my_bigrams_summ, my_bigrams_source)
    percent_new_bigrams.append(len(diff_bigram)/len(my_bigrams_summ))

    diff_trigram = np.setdiff1d(my_trigrams_summ, my_trigrams_source)
    percent_new_trigrams.append(len(diff_trigram)/len(my_trigrams_summ))

    diff_4gram = np.setdiff1d(my_4grams_summ, my_4grams_source)
    percent_new_4grams.append(len(diff_4gram)/len(my_4grams_summ))


unigrams_percent = np.array(percent_new_unigrams)
print(f"the average percent of novel unigrams in the summary:\
    {np.mean(unigrams_percent)}\n")
print(f"the std in the percent of novel unigrams in the summary: \
    {np.std(unigrams_percent)}\n")

bigrams_percent = np.array(percent_new_bigrams)
print(f"the average percent of novel bigrams in the summary: {np.mean(bigrams_percent)}\n")
print(f"the std in the percent of novel bigrams in the summary: {np.std(bigrams_percent)}\n")

trigrams_percent = np.array(percent_new_trigrams)
print(f"the average percent of novel trigrams in the summary: {np.mean(trigrams_percent)}\n")
print(f"the std in the percent of novel trigrams in the summary: {np.std(trigrams_percent)}\n")

four_grams_percent = np.array(percent_new_4grams)
print(f"the average percent of novel 4grams in the summary: {np.mean(four_grams_percent)}\n")
print(f"the std in the percent of novel 4grams in the summary: {np.std(four_grams_percent)}\n")
