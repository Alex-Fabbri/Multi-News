#----------------------------------------------------------------------------------
# Description:	Sentence class to store setences from the individual files in the
#				document cluster.
#----------------------------------------------------------------------------------

from nltk.corpus import stopwords

class sentence(object):

	#------------------------------------------------------------------------------
	# Description	: Constructor to initialize the setence object
	# Parameters  	: docName, name of the document/file
	#				  preproWords, words of the file after the stemming process
	#				  originalWords, actual words before stemming
	# Return 		: None
	#------------------------------------------------------------------------------
	# def __init__(self, docName, preproWords, originalWords):
	def __init__(self, preproWords, originalWords):
		# self.docName = docName
		self.preproWords = preproWords
		self.wordFrequencies = self.sentenceWordFreq()
		self.originalWords = originalWords

	#------------------------------------------------------------------------------
	# Description	: Function to return the name of the document
	# Parameters	: None
	# Return 		: name of the document
	#------------------------------------------------------------------------------
	def getDocName(self):
		# return self.docName
		pass
	
	#------------------------------------------------------------------------------
	# Description	: Function to return the stemmed words
	# Parameters	: None
	# Return 		: stemmed words of the sentence
	#------------------------------------------------------------------------------
	def getPreProWords(self):
		return self.preproWords
	
	#------------------------------------------------------------------------------
	# Description	: Function to return the original words of the sentence before
	#				  stemming
	# Parameters	: None
	# Return 		: pre-stemmed words
	#------------------------------------------------------------------------------
	def getOriginalWords(self):
		return self.originalWords

	#------------------------------------------------------------------------------
	# Description	: Function to return a dictonary of the word frequencies for
	#				  the particular sentence object
	# Parameters	: None
	# Return 		: dictionar of word frequencies
	#------------------------------------------------------------------------------
	def getWordFreq(self):
		return self.wordFrequencies	
	
	#------------------------------------------------------------------------------
	# Description	: Function to create a dictonary of word frequencies for the
	#				  sentence object
	# Parameters	: None
	# Return 		: dictionar of word frequencies
	#------------------------------------------------------------------------------
	def sentenceWordFreq(self):
		wordFreq = {}
		for word in self.preproWords:
			if word not in wordFreq.keys():
				wordFreq[word] = 1
			else:
				# if word in stopwords.words('english'):
				# 	wordFreq[word] = 1
				# else:			
				wordFreq[word] = wordFreq[word] + 1
		return wordFreq