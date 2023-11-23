import json
import string
from typing import Generator
from pathlib import Path
from dataclasses import dataclass

import spacy
import nltk
import tqdm
from nltk.corpus import stopwords

import os
import re
import zipfile


@dataclass
class Document:
    name: str
    tokens: list[str]


class PostPdfDocReader:
    def __init__(self):
        self.path = os.path.join(os.getcwd(), 'data', 'archive.zip')
        self.archive = zipfile.ZipFile(self.path, 'r')
        self.tokenizer = nltk.RegexpTokenizer(r'[^\d\W]{2}[^\d\W]+[-]*[^\d\W]*')

        nltk.download('stopwords')
        self.stop_words = stopwords.words('russian')
        self.stop_words.extend(string.punctuation + string.whitespace)
        self.not_empty_line_filter = (lambda line: len(line) > 0)
        self.stop_word_filter = (lambda word: word not in self.stop_words)
        
        self.nlp = spacy.load('ru_core_news_md', disable=['parser', 'ner'])
        self._lemma_cache = {}

    def _get_token_lemma(self, token: str) -> str:
        if token not in self._lemma_cache:
            lemma = self.nlp(token)[0].lemma_
            self._lemma_cache[token] = lemma

        return self._lemma_cache[token]

    def _read_doc_lines(self, lines: list[str]) -> list[list[str]]:
        lines = [self.tokenizer.tokenize(l.decode("utf-8").strip().lower()) for l in lines]
        filled_lines = list(filter(self.not_empty_line_filter, lines))
        lemmatized_lines = [[self._get_token_lemma(token) for token in line] for line in filled_lines]
        return lemmatized_lines

    def _read_single_file(self, filename: str):
        if re.fullmatch(r'data.*elibrary_.*\.txt', filename) == None:
            return None, None, None
        category = re.search(r"(?<=/\d ).*(?=/)", filename)
        if category == None:
            category = re.search(r"(?<=/\d\d ).*(?=/)", filename).group()
        else:
            category = category.group()
        doc_id = re.search(r"(?<=elibrary_).*(?=.txt)", filename).group()
        doc_data = []
        with self.archive.open(filename) as file:
            lines = self._read_doc_lines(file.readlines())
            for line_idx, line in enumerate(lines):
                for token_idx, token in enumerate(line):
                    doc_data.append([token, token_idx, line_idx, doc_id, category])
        df = pd.DataFrame(doc_data, columns =['token', 'doc_token_id', 'doc_line_id', 'doc_id', 'category'])
        return df, category, doc_id

    def read_files(self):
        names = self.archive.namelist()
        for name in names:
            df, category, doc_id = self._read_single_file(name)
            if df is not None:
                yield df, category, doc_id
    
    def read_and_save(self, dirname: str):
        os.mkdir(dirname)
        names = self.archive.namelist()
        for name in tqdm(names):
            df, category, doc_id = self._read_single_file(name)
            if df is not None:
                df.to_csv(os.path.join(dirname, category + '-' + doc_id + '.csv'))


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
