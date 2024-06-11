import math
class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		s = set(true_doc_IDs)
		precision = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in s:
				precision += 1
		
		precision /= k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[int(i['query_num'])].append(int(i['id']))

		meanPrecision = 0
		for i in range(len(query_ids)):
			meanPrecision += self.queryPrecision(doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)

		meanPrecision /= len(query_ids)

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		s = set(true_doc_IDs)
		recall = 0
		for i in range(k):
			if query_doc_IDs_ordered[i] in s:
				recall += 1
		
		recall /= len(s)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[int(i['query_num'])].append(int(i['id']))

		meanRecall = 0
		for i in range(len(query_ids)):
			meanRecall += self.queryRecall(doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)

		meanRecall /= len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = 0

		if precision + recall != 0:
			fscore = 2 * precision * recall / (precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[int(i['query_num'])].append(int(i['id']))
		
		meanFscore = 0
		for i in range(len(query_ids)):
			meanFscore += self.queryFscore(doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)
		
		meanFscore /= len(query_ids)
		return meanFscore
	


	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""
		DCG = 0
		IDCG = 0

		for i in range(k):
			doc_id = query_doc_IDs_ordered[i]
			log_value = math.log(i + 2, 2)
			rel_value = true_doc_IDs.get(doc_id, 0)
			DCG = DCG + rel_value/log_value

		rel_values = []
		for docs in true_doc_IDs:
			rel_values.append(true_doc_IDs[docs])
		rel_values.sort(reverse=True)

		for i in range(min(k, len(rel_values))):
			log_value = math.log(i + 2, 2)
			rel_value = rel_values[i]
			IDCG = IDCG + rel_value / log_value

		if IDCG > 0:
			return DCG / IDCG
		
		else:
			return 0
	


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = 0
		for i in range(len(query_ids)):
			rel_docs = {}
			for rel in qrels:
				if int(rel["query_num"]) == query_ids[i]:
					rel_docs[int(rel["id"])] = 5 -	rel["position"] 
			meanNDCG += self.queryNDCG(doc_IDs_ordered[i], query_ids[i], rel_docs, k)
		meanNDCG /= len(query_ids)
		return meanNDCG



	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = 0
		count_rel = 0
		for i in range(1,k+1):
			if query_doc_IDs_ordered[i - 1] in true_doc_IDs:
				count_rel += 1
				avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
		
		if count_rel == 0:
			return 0
		else:
			avgPrecision = avgPrecision / count_rel

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		rel={}
		for i in query_ids:
			rel[i]=[]
		for i in qrels:
			rel[int(i['query_num'])].append(int(i['id']))
		
		meanAveragePrecision = 0
		for i in range(len(query_ids)):
			meanAveragePrecision += self.queryAveragePrecision(doc_IDs_ordered[i], query_ids[i], rel[query_ids[i]], k)
		
		meanAveragePrecision /= len(query_ids)

		return meanAveragePrecision
	