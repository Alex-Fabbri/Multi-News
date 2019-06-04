import nltk
import os
import math
import string
import re
import sentence
import copy
import sys
from nltk.corpus import stopwords
from gensim.summarization.textcleaner import split_sentences

counter_sim_devide0 = 0 
sentence_sim_exception = 0.000000000000000000001

def load_data(data_path, file_name):
	counter = 0
	dataset = []
	with open(data_path + file_name,'r') as fr:
		for line in fr:
			dataset.append(line.strip())
			counter += 1

	print("[Data Loaded]: %s lines has been loaded." % counter)
	return dataset
#---------------------------------------------------------------------------------
# Description	: Function to preprocess the files in the document cluster before
#				  passing them into the MMR summarizer system. Here the sentences
#				  of the document cluster are modelled as sentences after extracting
#				  from the files in the folder path. 
# Parameters	: file_name, name of the file in the document cluster
# Return 		: list of sentence object
#---------------------------------------------------------------------------------
def processFile(sample):

	# read file from provided folder path
	# f = open(file_name,'r')
	# text_0 = f.read()
	text_0 = sample

	# extract content in TEXT tag and remove tags
	# text_1 = re.search(r"<TEXT>.*</TEXT>",text_0, re.DOTALL)
	# text_1 = re.sub("<TEXT>\n","",text_1.group(0))
	# text_1 = re.sub("\n</TEXT>","",text_1)

	# # replace all types of quotations by normal quotes
	# text_1 = re.sub("\n"," ",text_1)
	
	# text_1 = re.sub("\"","\"",text_1)
	# text_1 = re.sub("''","\"",text_1)
	# text_1 = re.sub("``","\"",text_1)	
	
	# text_1 = re.sub(" +"," ",text_1)

	# segment data into a list of sentences
	# sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
	# lines = sentence_token.tokenize(text_1.strip())	
	lines = split_sentences(text_0 + "\n")

	# setting the stemmer
	sentences = []
	porter = nltk.PorterStemmer()

	# modelling each sentence in file as sentence object
	for line in lines:

		# original words of the sentence before stemming
		originalWords = line[:]
		line = line.strip().lower()

		# word tokenization
		sent = nltk.word_tokenize(line)
		
		# stemming words
		stemmedSent = [porter.stem(word) for word in sent]		
		# stemmedSent = filter(lambda x: x!='.'and x!='`'and x!=','and x!='?'and x!="'" 
		# 	and x!='!' and x!='''"''' and x!="''" and x!="'s", stemmedSent)
		
		# list of sentence objects
		if stemmedSent != []:
			# sentences.append(sentence.sentence(file_name, stemmedSent, originalWords))	
			sentences.append(sentence.sentence(stemmedSent, originalWords))			
	
	return sentences

#---------------------------------------------------------------------------------
# Description	: Function to find the term frequencies of the words in the
#				  sentences present in the provided document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, term frequency score
#---------------------------------------------------------------------------------
def TFs(sentences):
	# initialize tfs dictonary
	tfs = {}

	# for every sentence in document cluster
	for sent in sentences:
		# retrieve word frequencies from sentence object
		wordFreqs = sent.getWordFreq()
	    
		# for every word
		for word in wordFreqs.keys():
			# if word already present in the dictonary
			if tfs.get(word, 0) != 0:				
				tfs[word] = tfs[word] + wordFreqs[word]
	        # else if word is being added for the first time
			else:				
				tfs[word] = wordFreqs[word]	
	return tfs

#---------------------------------------------------------------------------------
# Description	: Function to find the inverse document frequencies of the words in
#				  the sentences present in the provided document cluster 
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, inverse document frequency score
#---------------------------------------------------------------------------------
def IDFs(sentences):
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}
    w2 = []
    
    # every sentence in our cluster
    for sent in sentences:
        # print("In IDFs sent.getPreProWords() preproWords", sent.preproWords)
        # every word in a sentence
        for word in sent.getPreProWords():
            
            # not to calculate a word's IDF value more than once
            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0)+ 1

    # for each word in words
    for word in words:
        n = words[word]
        
        # avoid zero division errors
        try:
            w2.append(n)
            idf = math.log10(float(N)/n)
        except ZeroDivisionError:
            idf = 0
                
        # reset variables
        idfs[word] = idf
            
    return idfs

#---------------------------------------------------------------------------------
# Description	: Function to find TF-IDF score of the words in the document cluster
# Parameters	: sentences, sentences of the document cluster
# Return 		: dictonary of word, TF-IDF score
#---------------------------------------------------------------------------------
def TF_IDF(sentences):
    # Method variables
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}

    # for every word
    for word in tfs:
        #calculate every word's tf-idf score
        try:
            tf_idfs=  tfs[word] * idfs[word]
            # print("OK!", word)
        except KeyError:
        	print("No OK", word)
        
        # add word and its tf-idf score to dictionary
        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval

#---------------------------------------------------------------------------------
# Description	: Function to find the sentence similarity for a pair of sentences
#				  by calculating cosine similarity
# Parameters	: sentence1, first sentence
#				  sentence2, second sentence to which first sentence has to be compared
#				  IDF_w, dictinoary of IDF scores of words in the document cluster
# Return 		: cosine similarity score
#---------------------------------------------------------------------------------
def sentenceSim(sentence1, sentence2, IDF_w):
	numerator = 0
	denominator = 0	
	# print("sentence to compare",sentence1.originalWords, sentence2.originalWords)
	for word in sentence2.getPreProWords():		
		numerator+= sentence1.getWordFreq().get(word,0) * sentence2.getWordFreq().get(word,0) *  IDF_w.get(word,0) ** 2

	for word in sentence1.getPreProWords():
		denominator+= ( sentence1.getWordFreq().get(word,0) * IDF_w.get(word,0) ) ** 2

	# check for divide by zero cases and return back minimal similarity
	try:
		return numerator / math.sqrt(denominator)
	except ZeroDivisionError:
		# return float("-inf")
		return sentence_sim_exception

#---------------------------------------------------------------------------------
# Description	: Function to build a query of n words on the basis of TF-IDF value
# Parameters	: sentences, sentences of the document cluster
#				  IDF_w, IDF values of the words
#				  n, desired length of query (number of words in query)
# Return 		: query sentence consisting of best n words
#---------------------------------------------------------------------------------
def buildQuery(sentences, TF_IDF_w, n):
	#sort in descending order of TF-IDF values
	scores = TF_IDF_w.keys()
	scores = sorted(scores, reverse=True)	
	
	i = 0
	j = 0
	queryWords = []

	# print("n, len(scores):", n, len(scores))
	# if len(scores) == 1:
	# 	print(TF_IDF_w)
	# select top n words
	while(i<n):
		words = TF_IDF_w[scores[j]]
		for word in words:
			queryWords.append(word)
			i=i+1
			if (i>n): 
				break
		j=j+1

	# return the top selected words as a sentence
	# return sentence.sentence("query", queryWords, queryWords)
	return sentence.sentence(queryWords, queryWords)

#---------------------------------------------------------------------------------
# Description	: Function to find the best sentence in reference to the query
# Parameters	: sentences, sentences of the document cluster
#				  query, reference query
#				  IDF, IDF value of words of the document cluster
# Return 		: best sentence among the sentences in the document cluster
#---------------------------------------------------------------------------------
def bestSentence(sentences, query, IDF):
	best_sentence = None
	maxVal = float("-inf")

	for sent in sentences:
		similarity = sentenceSim(sent, query, IDF)		

		if similarity > maxVal:
			best_sentence = sent
			maxVal = similarity
	# print("query", query.originalWords)
	# if best_sentence == None:
		# print([ sent.originalWords for sent in sentences ])
	global counter_sim_devide0
	# print("maxVal", maxVal)
	if maxVal == sentence_sim_exception:
		print("Some errors")
		counter_sim_devide0 += 1
	sentences.remove(best_sentence)

	return best_sentence

#---------------------------------------------------------------------------------
# Description	: Function to create the summary set of a desired number of words 
# Parameters	: sentences, sentences of the document cluster
#				  best_sentnece, best sentence in the document cluster
#				  query, reference query for the document cluster
#				  summary_length, desired number of words for the summary
#				  labmta, lambda value of the MMR score calculation formula
#				  IDF, IDF value of words in the document cluster 
# Return 		: name 
#---------------------------------------------------------------------------------
def makeSummary(sentences, best_sentence, query, summary_length, lambta, IDF):	
	summary = [best_sentence]
	sum_len = len(best_sentence.getPreProWords())

	MMRval={}

	# keeping adding sentences until number of words exceeds summary length
	while (sum_len < summary_length and len(sentences) != 0):	
		MMRval={}		

		for sent in sentences:
			MMRval[sent] = MMRScore(sent, query, summary, lambta, IDF)

		maxxer = max(MMRval, key=MMRval.get)
		summary.append(maxxer)
		sentences.remove(maxxer)
		sum_len += len(maxxer.getPreProWords())	

	return summary

#---------------------------------------------------------------------------------
# Description	: Function to calculate the MMR score given a sentence, the query
#				  and the current best set of sentences
# Parameters	: Si, particular sentence for which the MMR score has to be calculated
#				  query, query sentence for the particualr document cluster
#				  Sj, the best sentences that are already selected
#				  lambta, lambda value in the MMR formula
#				  IDF, IDF value for words in the cluster
# Return 		: name 
#---------------------------------------------------------------------------------
def MMRScore(Si, query, Sj, lambta, IDF):	
	Sim1 = sentenceSim(Si, query, IDF)
	l_expr = lambta * Sim1
	value = [float("-inf")]

	for sent in Sj:
		Sim2 = sentenceSim(Si, sent, IDF)
		value.append(Sim2)

	r_expr = (1-lambta) * max(value)
	MMR_SCORE = l_expr - r_expr	

	# return MMRScore
	return l_expr - r_expr

# -------------------------------------------------------------
#	MAIN FUNCTION
# -------------------------------------------------------------
if __name__=='__main__':
	print("[USAGE]: python mmr_summarizer.py <dataset_path> <dataset_name> <output_file_name>")
	if len(sys.argv) == 4:
		corpus = load_data(sys.argv[1], sys.argv[2])

		list_of_summarization = []
		for i in range(len(corpus)):
			sample = corpus[i]

			sentences = processFile(sample)
				
			original_sentences = [ sent.originalWords for sent in sentences ]

			IDF_w 		= IDFs(sentences)
			TF_IDF_w 	= TF_IDF(sentences)
		# 	# build query; set the number of words to include in our query
			query = buildQuery(sentences, TF_IDF_w, 8)
		# 	# pick a sentence that best matches the query	
			best1sentence = bestSentence(sentences, query, IDF_w)
			# build summary by adding more relevant sentences
			summary = makeSummary(sentences, best1sentence, query, 100, 0.5, IDF_w)

			tmp_summary = []
			for sent in summary:
				original_sent = sent.getOriginalWords()
				tmp_summary.append(original_sent)

			list_of_summarization.append(" ".join(tmp_summary))

		with open(sys.argv[3], 'w') as fw:
			for sample in list_of_summarization:
				fw.write(sample + "\n")
	else:
		print("Please run the scripts with [USAGE]")