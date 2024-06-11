from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import json
import sys
import os

dir_path = os.path.dirname(__file__)
embeddings_index = {}
build_embeddings = len(sys.argv) > 1 and sys.argv[1] == 'build-index'
tokenizer = None

def get_cranfield_docs():
	with open(os.path.join(dir_path, 'cranfield/cran_docs.json')) as f:
		cran_docs = json.load(f)
	return cran_docs

def get_cranfield_queries():
	with open(os.path.join(dir_path, 'cranfield/cran_queries.json')) as f:
		cran_queries = json.load(f)
	return cran_queries

def flatten_docs(docs):
	flattened_docs = []
	for doc in docs:
		flattened_doc = []
		for sentence in doc:
			flattened_doc.extend(sentence)
		
		flattened_docs.append(flattened_doc)

	return flattened_docs

def flatten_queries(queries):
	flattened_queries = []
	for query in queries:
		flattened_query = []
		for sentence in query:
			flattened_query.extend(sentence)
		
		flattened_queries.append(flattened_query)

	return flattened_queries

def process_docs(docs):
	# from the Json, extract the title and body
	processed_docs = []
	for doc in docs:
		processed_docs.append( (doc['title'] + ' ' + doc['body']).lower() )

	return processed_docs
	
def process_queries(queries):
	processed_queries = []
	for query in queries:
		processed_queries.append(query['query'].lower())

	return processed_queries

def load_Glove_embeddings_pretrained():
	global embeddings_index

	with open(os.path.join(dir_path, 'models/glove.6B.50d.txt')) as f:
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:],dtype='float32')
			embeddings_index[word] = coefs
   
	return embeddings_index

def get_doc_embeddings(docs):
	global embeddings_index, tokenizer

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(docs)
	sequences = tokenizer.texts_to_sequences(docs)
	word_index = tokenizer.word_index
	inverted_word_index = {v: k for k, v in word_index.items()}

	doc_embeddings = []
	for sequence in sequences:
		doc_embedding = [embeddings_index[inverted_word_index[word_id]] for word_id in sequence if inverted_word_index[word_id] in embeddings_index]
		
		if doc_embedding:  # doc_embedding shdnt be empty
			doc_embedding = np.mean(doc_embedding, axis=0)
			doc_embeddings.append(doc_embedding)
		else:
			doc_embeddings.append(np.zeros(50))
	
	return doc_embeddings

def get_query_embeddings(queries):
	global embeddings_index, tokenizer

	sequences = tokenizer.texts_to_sequences(queries)
	word_index = tokenizer.word_index
	inverted_word_index = {v: k for k, v in word_index.items()}

	query_embeddings = []
	for sequence in sequences:
		query_embedding = [embeddings_index[inverted_word_index[word_id]] for word_id in sequence if inverted_word_index[word_id] in embeddings_index]
		
		if query_embedding:  # doc_embedding shdnt be empty
			query_embedding = np.mean(query_embedding, axis=0)
			query_embeddings.append(query_embedding)
		else:
			query_embeddings.append(np.zeros(50))
	
	return query_embeddings

def get_similar_docs(query, doc_embeddings):
	similarities = np.dot(doc_embeddings, query) / (np.linalg.norm(doc_embeddings) * np.linalg.norm(query))
	similarities = np.argsort(similarities)[::-1]

	return similarities

def rank_documents(docs, queries, build_embeddings):
	load_Glove_embeddings_pretrained()
	docs = flatten_docs(docs)
	queries = flatten_queries(queries)

	doc_embeddings = None
	query_embeddings = None
	if(build_embeddings):
		doc_embeddings = get_doc_embeddings(docs)
		np.save(os.path.join(dir_path, 'embeddings/doc_embeddings.npy'), doc_embeddings)
		print("Computed Doc Embeddings")
	else:
		doc_embeddings = np.load(os.path.join(dir_path, 'embeddings/doc_embeddings.npy'))

	if(build_embeddings):
		query_embeddings = get_query_embeddings(queries)
		np.save(os.path.join(dir_path, 'embeddings/query_embeddings.npy'), query_embeddings)
		print("Computed Query Embeddings")
	else:
		query_embeddings = np.load(os.path.join(dir_path, 'embeddings/query_embeddings.npy'))

	doc_ids_ordered = []
	for query in query_embeddings:
		doc_ids_ordered.append(get_similar_docs(query, doc_embeddings))

	return doc_ids_ordered

if __name__ == '__main__':
	load_Glove_embeddings_pretrained()

	docs = get_cranfield_docs()
	docs = process_docs(docs)

	if(build_embeddings):
		doc_embeddings = get_doc_embeddings(docs)
		np.save(os.path.join(dir_path, 'embeddings/doc_embeddings.npy'), doc_embeddings)

	queries = get_cranfield_queries()
	queries = process_queries(queries)

	if(build_embeddings):
		query_embeddings = get_query_embeddings(queries)
		np.save(os.path.join(dir_path, 'embeddings/query_embeddings.npy'), query_embeddings)


	