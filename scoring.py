import tqdm
import numpy as np

from octis.evaluation_metrics.coherence_metrics import Coherence
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from data_reader import Document


def get_topics_ctfidf(
        components: list[Document],
        top_k=20,
        min_df=10,
        max_df=0.75,
        reduce_frequent_words=False,
        bm25_weighting=False,
) -> dict[str, list[str]]:
    texts_sentences = [' '.join(component.tokens) for component in components]
    count_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
    sentences_cv = count_vectorizer.fit_transform(texts_sentences)

    ctfidf = ClassTfidfTransformer(bm25_weighting, reduce_frequent_words)
    sentences_ctfidf = ctfidf.fit_transform(sentences_cv)

    component_name_to_topics = {}
    for component, row in tqdm.tqdm(zip(components, sentences_ctfidf)):
        most_scored_idx = np.argsort(-row.toarray())[0, :top_k]
        topic_words = count_vectorizer.get_feature_names_out()[most_scored_idx]
        component_name_to_topics[component.name] = list(topic_words)

    return component_name_to_topics
