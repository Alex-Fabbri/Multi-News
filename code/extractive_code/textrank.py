from gensim.summarization.summarizer import summarize
from gensim.summarization.textcleaner import split_sentences
import sys

def read_in_train_set(input_path, filename):
	corpus = []
	with open(input_path + filename, 'r') as fr:
		for line in fr:
			corpus.append(line.strip())
	return corpus

def textrank_summarize(corpus):
	print("Begin summarizing...")

	list_of_summarization = []

	error_counter = 0
	null_summarization_counter = 0
	for i in range(len(corpus)):
		sample = corpus[i].strip()
		articles = sample.split("story_separator_special_tag")
		
		try:
			summarization = summarize("\n".join(articles), word_count = 500, split = True)
			if len(summarization) == 0:
				null_summarization_counter += 1
				summarization = split_sentences("\n".join(articles))
				if len(summarization) == 0:
					print("*** No Summarization ***", i)	
		except ValueError:
			print("ValueError, sample", sample)
			summarization = sample
			list_of_summarization.append(summarization)
			error_counter += 1
			continue

		tmp_list_of_summarization = [ [] for _ in range(len(articles)) ]
		for sent in summarization:
			flag = 0
			for j in range(len(articles)):
				if sent in articles[j]:
					tmp_list_of_summarization[j].append(sent)
					flag = 1
			if flag == 0:
				print(i, "****", sent, (sent in " ".join(articles)))
			
		for k in range(len(tmp_list_of_summarization)):
			tmp_list_of_summarization[k] = " newline_char ".join(tmp_list_of_summarization[k])

		list_of_summarization.append(" story_separator_special_tag ".join(tmp_list_of_summarization))

		if i %100 == 0:
			print(i)
			print("------")
		# if i == 5000:
		# 	break

	return list_of_summarization, error_counter, null_summarization_counter

if __name__ =='__main__':
	print("[USAGE]: python textrank.py <dataset_path> <dataset_name> <output_file_name>")
	if len(sys.argv) == 4:
		input_path = sys.argv[1]

		corpus = read_in_train_set(input_path, sys.argv[2])
		print("length", len(corpus))
		res, error, null_summarization_counter = textrank_summarize(corpus)

		print("Number of Summarization Error:", error)
		print("Number of Null Summarization:", null_summarization_counter)
		with open(sys.argv[3], 'w') as fw:
			for sample in res:
				fw.write(sample + "\n")
	else:
		print("Please run the scripts with [USAGE]")			