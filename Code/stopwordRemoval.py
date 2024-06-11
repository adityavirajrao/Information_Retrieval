import math
import nltk
import json
# Add your import statements here
from nltk.corpus import stopwords
class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = []
		#Fill in code here
		stop_words = set(stopwords.words('english'))
		for sentence_tokens in text:
			filtered_sentence = [token for token in sentence_tokens if token.lower() not in stop_words]
			stopwordRemovedText.append(filtered_sentence)
		return stopwordRemovedText


	def stopwords_from_corpus(self, text):
		docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]
		corpus = [item["body"] for item in docs_json]
		N = len(corpus)
		word_count = {}
		for document in corpus:
			document_sentences = nltk.sent_tokenize(document)
			words = set()
			for sentence in document_sentences:
				words.update(nltk.word_tokenize(sentence))
			for word in words:
				if word in word_count:
					word_count[word] += 1
				else:
					word_count[word] = 1
		idf_values = {}
		for word, count in word_count.items():
			idf_values[word] = math.log(N/count)

		threshold = 0.9 # change the threshold here
		stopwordsList = [word for word, idf in idf_values.items() if (idf < threshold and (word != '.' and word != ","))]
		stopwordRemovedText = []

		print(stopwordsList) # prints the stopwords from the corpus
		for sentence_tokens in text:
			filtered_sentence = [token for token in sentence_tokens if token.lower() not in stopwordsList]
			stopwordRemovedText.append(filtered_sentence)
		return stopwordRemovedText
