import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LSA():
    def __init__(self) -> None:
        self.doc_ids = None
        self.vectorizer = None
        self.svd = None
        self.reduced_matrix = None

    def buildIndex(self, documents, document_ids, ngram_range=(1, 1), concepts=0):
        all_docs_combined = []
        for document in documents:
            all_sentences_combined = []
            for sentence in document:
                all_sentences_combined.extend(sentence)
            all_docs_combined.append(all_sentences_combined)
        all_docs_combined = [' '.join(sentence) for sentence in all_docs_combined]
        tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        term_document_matrix = tfidf_vectorizer.fit_transform(all_docs_combined)
        svd_model = TruncatedSVD(random_state=42, n_components=concepts)
        reduced_matrix = svd_model.fit_transform(term_document_matrix)

        self.doc_ids = document_ids
        self.vectorizer = tfidf_vectorizer
        self.svd = svd_model
        self.reduced_matrix = reduced_matrix

    def rank(self, queries):
        ranked_doc_ids = []
        all_queries_combined = []
        for query in queries:
            all_sentences_combined = []
            for sentence in query:
                all_sentences_combined.extend(sentence)
            all_queries_combined.append(all_sentences_combined)
        all_queries_combined = [' '.join(sentence) for sentence in all_queries_combined]

        query_vectorizer = self.vectorizer.transform(all_queries_combined)
        query_vectors = self.svd.transform(query_vectorizer)

        similarity_values = cosine_similarity(query_vectors, self.reduced_matrix)
        for value in similarity_values:
            scores = []
            total_docs = len(self.doc_ids)
            for i in range(total_docs):
                scores.append((self.doc_ids[i], value[i]))
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            ordered_docs_query = []
            for i in sorted_scores:
                ordered_docs_query.append(i[0])
            ranked_doc_ids.append(ordered_docs_query)
        return ranked_doc_ids
