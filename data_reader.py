import json
import string
from typing import Generator
from pathlib import Path
from dataclasses import dataclass

import spacy
import nltk
import tqdm
from nltk.corpus import stopwords


@dataclass
class Document:
    name: str
    tokens: list[str]


class JsonDocReader:
    def __init__(
            self,
            path: Path,
            filter_stopwords: bool = True,
            lemmatize: bool = True,
            keep_only_words: bool = True,
            min_token_len: int = 4,
    ):
        self.path = path
        self.filter_stopwords = filter_stopwords
        self.lemmatize = lemmatize
        self.keep_only_words = keep_only_words
        self.min_token_len = min_token_len

        self.tokenizer = nltk.WordPunctTokenizer()
        self.filters = []

        if self.filter_stopwords:
            nltk.download('stopwords')
            self.stop_words = stopwords.words('russian')
            self.stop_words.extend(string.punctuation + string.whitespace)
            self.filters.append(lambda word: word not in self.stop_words)

        if self.keep_only_words:
            self.filters.append(lambda word: word.isalpha())

        if self.lemmatize:
            self.nlp = spacy.load('ru_core_news_md', disable=['parser', 'ner'])
            self._lemma_cache = {}

        self.filters.append(lambda word: len(word) > self.min_token_len)

    def _read_text(self) -> Generator[tuple[str, str], None, None]:
        with open(self.path, 'r') as f:
            docs = json.load(f)

        for name in docs:
            yield name, docs[name]

    def _get_token_lemma(self, token: str) -> str:
        if token not in self._lemma_cache:
            lemma = self.nlp(token)[0].lemma_
            self._lemma_cache[token] = lemma

        return self._lemma_cache[token]

    def _apply_filters(self, tokens: list[str]) -> list[str]:
        for filter_ in self.filters:
            tokens = filter(filter_, tokens)

        return list(tokens)

    def _lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        return [self._get_token_lemma(token) for token in tokens]

    def read_documents(self) -> Generator[Document, None, None]:
        text_reader = self._read_text()
        for name, text in tqdm.tqdm(text_reader):
            tokens = self.tokenizer.tokenize(text)
            tokens = self._apply_filters(tokens)
            if self.lemmatize:
                tokens = self._lemmatize_tokens(tokens)

            yield Document(name, tokens)
