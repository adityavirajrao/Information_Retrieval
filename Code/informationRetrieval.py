from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain

class InformationRetrieval():

    def __init__(self):
        self.index = None

    def buildIndex(self, docs, docIDs, ngram_range=(1,1), concepts=0):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable
        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        all_docs_combined = []
        for document in docs:
            all_sentences_combined = []
            for sentence in document:
                all_sentences_combined.extend(sentence)
            all_docs_combined.append(all_sentences_combined)
        all_docs_combined = [' '.join(sentence) for sentence in all_docs_combined]
        self.term_doc_freq = self.tfidf_vectorizer.fit_transform(all_docs_combined)
        self.docIDs = docIDs
        self.index = self.term_doc_freq.T

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query
        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        
        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """

        doc_IDs_ordered = []

        # Fill in code here
        term_doc_freq = self.term_doc_freq
        self.tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=False, sublinear_tf=False)
        self.tfidf_transformer.fit(term_doc_freq)
        tfidf = self.tfidf_transformer.transform(term_doc_freq)
        all_queries_combined = []
        for query in queries:
            all_sentences_combined = []
            for sentence in query:
                all_sentences_combined.extend(sentence)
            all_queries_combined.append(all_sentences_combined)
        all_queries_combined = [' '.join(sentence) for sentence in all_queries_combined]
        query_vectorizer = self.tfidf_vectorizer.transform(all_queries_combined)
        query_vectors = self.tfidf_transformer.transform(query_vectorizer)
        similarity_values = cosine_similarity(query_vectors, tfidf)
        for value in similarity_values:
            scores = []
            total_docs = len(self.docIDs)
            for i in range(total_docs):
                scores.append((self.docIDs[i], value[i]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            ordered_docs_query = []
            for i in sorted_scores:
                ordered_docs_query.append(i[0])
            doc_IDs_ordered.append(ordered_docs_query)
        return doc_IDs_ordered
    