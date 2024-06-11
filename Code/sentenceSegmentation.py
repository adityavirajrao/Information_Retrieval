import nltk
print("NLTK Data Path: \n", nltk.data.path)

nltk.download('punkt')
print("downloaded punkt\n\n")

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		# split the text into segments at punctuation marks that typically signal sentence endings 
		# like periods (.), question marks (?), and exclamation points (!). 

		str_copy = str.replace(text, "!", ".")
		str_copy = str.replace(str_copy, "?", ".")

		segmentedText = str_copy.split(".")

		return segmentedText

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		# use the Punkt tokenizer to segment the text into sentences
		sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
		segmentedText = sentence_detector.tokenize(text.strip())
		
		return segmentedText