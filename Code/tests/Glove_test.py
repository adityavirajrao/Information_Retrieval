import re
import os
import sys
dir_path_local = os.path.dirname(__file__)
# adding the source directory to the PYTHONPATH env variable
sys.path.append(os.path.join(dir_path_local, '..'))

import numpy as np
from Glove import *
from sentenceSegmentation import SentenceSegmentation

def test_precomputed_embeddings():
    doc_embeddings = np.load(os.path.join(dir_path_local, '../embeddings/doc_embeddings.npy'))
    assert doc_embeddings.shape == (1400, 50)

    query_embeddings = np.load(os.path.join(dir_path_local, '../embeddings/query_embeddings.npy'))    
    assert query_embeddings.shape == (225, 50)

def test_get_doc_embeddings():
    embeddings_index = load_Glove_embeddings_pretrained()

    docs = ['test', 'hello']
    doc_embeddings = get_doc_embeddings(docs)

    for i in range(len(docs)):
        assert np.array_equal(doc_embeddings[i], embeddings_index[docs[i]])
        
def test_avg_sen_len_cranfield():
    cran_docs = get_cranfield_docs()
    cran_queries = get_cranfield_queries()

    total_sentences = 0
    for doc in cran_docs:
        doc = doc['body'] 
        segmented_doc = SentenceSegmentation().punkt(doc)
        total_sentences += len(segmented_doc)
    
    total_sentences_query = 0
    for query in cran_queries:
        query = query['query']
        segmented_query = SentenceSegmentation().punkt(query)
        total_sentences_query += len(segmented_query)
    
    avg_sen_len = total_sentences / len(cran_docs)
    avg_sen_len_query = total_sentences_query / len(cran_queries)

    print("Average Sentence Length in Cranfield Docs: ", avg_sen_len)
    print("Average Sentence Length in Cranfield Queries: ", avg_sen_len_query)

def test_author_in_cranfield_queries():
    docs = get_cranfield_docs()
    authors = [doc['author'] for doc in docs]

    queries = get_cranfield_queries()
    queries = [query['query'] for query in queries]

    distinct_tokens = set()
    for author in authors:
        tokens = re.split('[ .,]', author)
        tokens = [' ' + token.lower() + ' ' for token in tokens if len(token) > 2 and token != "and"]

        for token in tokens:
            for query in queries:
                if token in query:
                    print("Author token found: ", token)
                    print("Query: ", query)
                    distinct_tokens.add(token)
    

    print("Distinct author tokens found in queries: ", distinct_tokens)