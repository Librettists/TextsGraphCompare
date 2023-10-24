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
    def __init__(self, path: Path, filter_stopwords: bool = True, lemmatize: bool = True):
        self.path = path
        self.filter_stopwords = filter_stopwords
        self.lemmatize = lemmatize

        self.tokenizer = nltk.WordPunctTokenizer()

        if self.filter_stopwords:
            nltk.download('stopwords')
            self.stop_words = stopwords.words('russian')
            self.stop_words.extend(string.punctuation + string.whitespace)

        if self.lemmatize:
            self.nlp = spacy.load('ru_core_news_md', disable=['parser', 'ner'])
            self._lemma_cache = {}

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

    def _filter_stopwords(self, tokens: list[str]) -> list[str]:
        return list(filter(lambda token: token not in self.stop_words, tokens))

    def _lemmatize_doc(self, text: str) -> list[str]:
        return [self._get_token_lemma(token) for token in self.tokenizer.tokenize(text)]

    def read_documents(self) -> Generator[Document, None, None]:
        text_reader = self._read_text()
        for name, text in tqdm.tqdm(text_reader):
            if self.lemmatize:
                tokens = self._lemmatize_doc(text)
            else:
                tokens = self.tokenizer.tokenize(text)

            if self.filter_stopwords:
                tokens = self._filter_stopwords(tokens)

            yield Document(name, tokens)
