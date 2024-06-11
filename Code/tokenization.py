import nltk

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = []

		# Split the sentence into words based on whitespace (spaces, tabs, and newlines).
		for sentence in text:
			sentence.replace("\n", " ")
			sentence.replace("\t", " ")

			tokenizedText.append(sentence.split())

		return tokenizedText

	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		# Use the Penn Tree Bank Tokenizer to tokenize the text
		tokenizer = nltk.tokenize.TreebankWordTokenizer()
		tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in text]

		return tokenized_sentences