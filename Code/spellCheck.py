import json
import nltk
import re
import math


def compute_edit_distance(str1, str2, C_i, C_d, C_s):
    # print()
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = C_i * j
            elif j == 0:
                dp[i][j] = C_d * i
            elif str1[i - 1] == str2[j - 1]:
                dp[i][j] = min(dp[i - 1][j] + C_d, dp[i][j - 1] + C_i, dp[i - 1][j - 1])
            else:
                dp[i][j] = min(dp[i - 1][j] + C_d, dp[i][j - 1] + C_i, dp[i - 1][j - 1] + C_s)

    return dp[m][n]

def calculate_cosine_similarity(vector1, vector2):
    dot_product = sum(x * y for x, y in zip(vector1, vector2))

    magnitude1 = math.sqrt(sum(x ** 2 for x in vector1))
    magnitude2 = math.sqrt(sum(y ** 2 for y in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


def build_word_vectors():
    docs_json = json.load(open("cranfield/cran_docs.json", 'r'))[:]
    corpus = [item["body"] for item in docs_json]
    vocabulary = set()
    for document in corpus:
        words = (nltk.word_tokenize(document.lower()))
        vocabulary.update(words)
    pattern = re.compile(r'^[a-zA-Z]+$')
    alpha_words = [word for word in vocabulary if pattern.match(word)]
    word_vectors = {}
    for word in alpha_words:
        word_length = len(word)
        word_vector = [0]*676
        for i in range(word_length - 1):
            pos = (ord(word[i]) - ord('a')) * 26 + (ord(word[i + 1]) - ord('a'))
            word_vector[pos] += 1
        word_vectors[word] = word_vector

    return word_vectors




def find_typo_candidates(typo, word_vectors):
    typo_length = len(typo)
    typo_vector = [0] * 676
    for i in range(typo_length - 1):
        pos = (ord(typo[i]) - ord('a')) * 26 + (ord(typo[i + 1]) - ord('a'))
        typo_vector[pos] += 1

    similarity_values = []
    for word, word_vector in word_vectors.items():
        similarity_values.append((word, calculate_cosine_similarity(typo_vector, word_vector)))

    sorted_candidates = sorted(similarity_values, key=lambda x: x[1], reverse=True)

    top5_candidates = []
    for i in range(0, 5):
        top5_candidates.append((sorted_candidates[i][0], sorted_candidates[i][1]))
    return top5_candidates


typo_list = ['boundery', 'transiant', 'aerplain']
vocabulary_vectors = build_word_vectors()
top_candidates = {}
for typo in typo_list:
    top_candidates[typo] = find_typo_candidates(typo, vocabulary_vectors)
    print(f"Top 5 candidates and their cosine similarities for typo '{typo}':")
    min_distance = float('inf')
    for candidate in top_candidates[typo]:
        print(f"({candidate[0]}, {round(candidate[1], 2)})")

print("Cost of insertion: 2, cost of deletion: 2, cost of substitution: 3.")
for typo in typo_list:
    min_distance = float('inf')
    closest_candidate = None
    for candidate in top_candidates[typo]:
        distance = compute_edit_distance(candidate[0], typo, 2, 2, 3)
        if distance < min_distance:
            min_distance = distance
            closest_candidate = candidate
    print(f"Closest candidate for {typo}: {closest_candidate[0]} (Edit distance: {min_distance})")
