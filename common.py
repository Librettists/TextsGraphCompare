from collections import Counter, defaultdict
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import networkx as nx
from wordcloud import WordCloud

from data_reader import Document


def jaccard_sim(first_doc, second_doc, _words_cache) -> float:
    if first_doc.name not in _words_cache:
        _words_cache[first_doc.name] = set(first_doc.tokens)
    if second_doc.name not in _words_cache:
        _words_cache[second_doc.name] = set(second_doc.tokens)

    first_doc_words = _words_cache[first_doc.name]
    second_doc_words = _words_cache[second_doc.name]

    return len(first_doc_words.intersection(second_doc_words)) / len(first_doc_words.union(second_doc_words))


def get_graph(docs, threshold, sim_fn):
    graph = nx.Graph()
    for first_ix, first_doc in tqdm.tqdm(enumerate(docs)):
        for second_ix, second_doc in enumerate(docs[first_ix + 1:]):
            sim = sim_fn(first_doc, second_doc)
            if sim < threshold:
                continue
            graph.add_edge(first_doc.name, second_doc.name, weight=sim)
            graph.add_edge(second_doc.name, first_doc.name, weight=sim)

    return graph


def get_component_most_common_words(component, doc_name_to_doc, n=20):
    words_count = Counter()
    for doc_name in component:
        words_count.update(doc_name_to_doc[doc_name].tokens)

    return words_count.most_common(n=n)


def wordcloud_from_component(component, doc_name_to_doc):
    print(len(component))
    most_common = get_component_most_common_words(component, doc_name_to_doc)

    text = ' '.join([word for (word, count) in most_common])
    pprint(most_common)
    wordcloud = WordCloud().generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


def compute_top_words_sim(topics, model, topn=20):
    avg_sim = 0
    for topic in topics:
        vectors = [model[word] for word in topic if word in model]

        avg_component_sim = 0
        for i, vec1 in enumerate(vectors):
            for vec2 in vectors[i + 1:]:
                avg_component_sim += (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        avg_sim += avg_component_sim / (topn * (topn - 1) / 2)
    avg_sim /= len(topics)

    return avg_sim


def _calculate_df(documents):
    df = defaultdict(int)
    for doc in documents:
        unique_tokens = set(doc.tokens)
        for token in unique_tokens:
            df[token] += 1

    return dict(df)


def filter_common_words(documents: list[Document], min_freq: int, max_freq: float):
    assert 0 <= max_freq <= 1
    assert 0 <= min_freq <= 1

    df = _calculate_df(documents)

    filtered_docs = []
    for doc in tqdm.tqdm(documents):
        filtered_tokens = list(filter(
            lambda token: min_freq <= df[token] / len(documents) <= max_freq,
            doc.tokens
        ))
        filtered_docs.append(Document(name=doc.name, tokens=filtered_tokens))

    return filtered_docs


def cosine_sim(first_doc, second_doc, doc_to_vec):
    first = doc_to_vec[first_doc.name]
    second = doc_to_vec[second_doc.name]

    return first @ second / np.linalg.norm(first) / np.linalg.norm(second)
