import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
class InflectionReduction:

    def reduce(self, text):
        """
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

        reducedText1 = []
        porter = PorterStemmer()
        for sentence in text:
            modifiedSentence = [porter.stem(token) for token in sentence]
            reducedText1.append(modifiedSentence)
            
        reducedText2 = []
        lemmatizer = WordNetLemmatizer()
        for sentence in text:
            modifiedSentence = [lemmatizer.lemmatize(token) for token in sentence]
            reducedText2.append(modifiedSentence)
            
        return reducedText2
