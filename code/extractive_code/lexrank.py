from lexrank import STOPWORDS, LexRank
from gensim.summarization.textcleaner import split_sentences
import time
import sys

def read_in_train_set(input_path, filename):
	corpus = []
	with open(input_path + filename, 'r') as fr:
		for line in fr:
			corpus.append(line.strip())
	return corpus

def lexrank_summarize(corpus):
	list_of_summarization = []

	documents = [ split_sentences(sample.replace("story_separator_special_tag", "\n")) for sample in corpus ]
	print("[" + "Document Size: " + str(len(documents)) + "]")
	print("[" + time.strftime("%H:%M:%S", time.localtime()) + "]", "Begin building LexRank model...")	
	lxr = LexRank(documents, stopwords=STOPWORDS['en'])
	print("[" + time.strftime("%H:%M:%S", time.localtime()) + "]", "LexRank model successfully built...")

	for i in range(len(documents)):
		sample = documents[i]
		summary = lxr.get_summary(sample, summary_size=len(sample))
		articles = corpus[i].split("story_separator_special_tag")

		words_counter = 0
		summary_counter = 0
		tmp_summary = [ [] for _ in range(len(articles)) ]

		while words_counter < 500 and summary_counter < len(summary):
			flag = 0
			for j in range(len(articles)):
				if summary[summary_counter] in articles[j]:
					tmp_summary[j].append(summary[summary_counter])
					words_counter += len(summary[summary_counter].split(" "))
					flag = 1
			if flag == 0:
				print("[Error] Summary not in original sample.", summary[summary_counter], i)
			summary_counter += 1
			
		# print("words_counter, summary_counter, total summary", words_counter, summary_counter, len(summary))
		for k in range(len(tmp_summary)):
			tmp_summary[k] = " newline_char ".join(tmp_summary[k])
		list_of_summarization.append(" story_separator_special_tag ".join(tmp_summary))

		if i %100 == 0:
			print("------")
			print(i)
			print("------")
		# if i == 100:
		# 	break

	return list_of_summarization

if __name__ =='__main__':
	print("[USAGE]: python lexrank.py <dataset_path> <dataset_name> <output_file_name>")
	if len(sys.argv) == 4:
		input_path = sys.argv[1]
		corpus = read_in_train_set(input_path, sys.argv[2])
		res = lexrank_summarize(corpus)

		with open(sys.argv[3], 'w') as fw:
			for sample in res:
				fw.write(sample + "\n")
	else:
		print("Please run the scripts with [USAGE]")