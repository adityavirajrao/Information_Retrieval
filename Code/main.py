from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from QueryExpansion import QueryExpansion
from evaluation import Evaluation
from LSA import LSA
from Glove import rank_documents
from sys import version_info
import argparse
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from spellchecker import SpellChecker

spell = SpellChecker()
K_val = 10

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")

class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()
		self.ngram = None
		self.concepts = int(self.args.concepts)

		self.MAPs = []
		self.nDCGs = []
		self.precisions = []
		self.recalls = []
		self.fscores = []

		ngram = self.args.ngram
		if ngram == "unigram":
			ngram = (1,1)
		elif ngram == "bigram":
			ngram = (2,2)
		elif ngram == "hybrid":
			ngram = (1,2)

		self.ngram = ngram
		if self.args.method == "lsa":
			self.informationRetriever = LSA()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)

	def spellCheckQueries(self, queries):
		if self.args.spellcheck == "False":
			return queries
		
		return [[[spell.correction(token) for token in sentence] for sentence in query] for query in queries]
	
	def expandQueries(self, queries):
		"""
		Call the required query expansion method
		"""
		if self.args.qexpand == "True":
			queries = QueryExpansion(queries)
		return queries

	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs

	def plotMetricsByConcepts(self):
		concepts_range = range(100, 1500, 100)
		MAPs, nDCGs, precisions, recalls, fscores = [], [], [], [], []

		for concepts in concepts_range:
			self.concepts = concepts
			self.evaluateDataset()
			
			# Store evaluation metrics
			MAPs.append(self.MAPs[-1])
			nDCGs.append(self.nDCGs[-1])
			precisions.append(self.precisions[-1])
			recalls.append(self.recalls[-1])
			fscores.append(self.fscores[-1])

		# Plotting
		plt.plot(concepts_range, precisions, label="Precision")
		plt.plot(concepts_range, recalls, label="Recall")
		plt.plot(concepts_range, fscores, label="F-score")
		plt.plot(concepts_range, MAPs, label="MAP")
		plt.plot(concepts_range, nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics vs. Concepts")
		plt.xlabel("Number of Concepts")
		plt.ylabel("Metric Value")
		plt.xticks(np.arange(100, 1500, 200))

		plt.savefig("plots/metrics_vs_concepts_LSA.png")
		plt.close()

	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		start_time = time.time()

		# Read queries
		queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
		query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)

		# spellcheck queries, default False
		# Tt takes > 5mins, and the queries do not have any spelling errors
		processedQueries = self.spellCheckQueries(processedQueries) 

		# Expand queries
		processedQueries = self.expandQueries(processedQueries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
          [(item["body"] + ". " + ". ".join([item["title"]] * 3)) for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		if(self.args.method != "glove"):
			# Build document index
			self.informationRetriever.buildIndex(processedDocs, doc_ids, self.ngram, self.concepts)
			# Rank the documents for each query
			doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
		else:
			doc_IDs_ordered = rank_documents(processedDocs, processedQueries, self.args.build_embeddings=="True")

		# Read relevance judements
		qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		self.precisions, self.recalls, self.fscores, self.MAPs, self.nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			self.precisions.append(precision)
			recall = self.evaluator.meanRecall(
				doc_IDs_ordered, query_ids, qrels, k)
			self.recalls.append(recall)
			fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, query_ids, qrels, k)
			self.fscores.append(fscore)
			print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
			MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, query_ids, qrels, k)
			self.MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, query_ids, qrels, k)
			self.nDCGs.append(nDCG)
			print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))
		
		end_time = time.time()

		ir_time = end_time - start_time
		print(f"Information Retrieval process took {ir_time} seconds.")

		# Plot the metrics and save plot 
		plt.plot(range(1, K_val+1), self.precisions, label="Precision")
		plt.plot(range(1, K_val+1), self.recalls, label="Recall")
		plt.plot(range(1, K_val+1), self.fscores, label="F-Score")
		plt.plot(range(1, K_val+1), self.MAPs, label="MAP")
		plt.plot(range(1, K_val+1), self.nDCGs, label="nDCG")
		plt.legend()
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")

		plt.savefig("plots/eval_plot.png")
		plt.close()

		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()

		# spellcheck queries, always True (not recommended for cranfield queries, only for custom queries)
		self.args.spellcheck = "True"
		processedQuery = self.spellCheckQueries([processedQuery])[0]

		# Process query
		processedQuery = self.preprocessQueries([query])[0]

		# Expand query
		processedQuery = self.expandQueries([processedQuery])[0]

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document index
		self.informationRetriever.buildIndex(processedDocs, doc_ids)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id_ in doc_IDs_ordered[:5]:
			print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-method',
                      default="lsa",
                      help="IDF | lsa | glove")
	parser.add_argument('-ngram',
                      default="hybrid",
                      help="unigram|bigram|hybrid")
	parser.add_argument('-concepts',
                      default= "300",
                      help="concepts used by lsa")
	parser.add_argument('-qexpand',
					  default= "True",
					  help="Perform Query Expansion")
	parser.add_argument('-spellcheck',
					  default= "False",
					  help="Perform Spellcheck, not recommended unless used with custom queries, cranfiled queries do not have spelling errors")
	parser.add_argument('-build_embeddings',
					  default= "True",
					  help="Build Glove Embeddings")
	parser.add_argument('-plot_concepts_graph',
					  default= "False",
					  help="Plot Evaluation Metrics vs Concepts graph")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()

	if args.plot_concepts_graph == "True":
		searchEngine.plotMetricsByConcepts()
