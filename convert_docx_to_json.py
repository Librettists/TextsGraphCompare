import os
import zipfile, re, docx
import json
import tqdm
from pathlib import Path


def _read_docx_from_zipfile(path):
    archive = zipfile.ZipFile(path, 'r')

    for name in archive.namelist():
        if not re.fullmatch(r'.*\.docx', name):
            continue

        with archive.open(name) as file:
            doc = docx.Document(file)
            
            text = '.'.join([' '.join(p.text.split()) for p in doc.paragraphs if len(p.text) > 0])

        yield name, text


docs = {}
# set your path to a zipfile
zipfile_path = Path().cwd() / 'data' / 'docx35.zip'
for name, text in tqdm.tqdm(_read_docx_from_zipfile(zipfile_path)):
    doc_name = Path(name).with_suffix('.txt').name
    docs[doc_name] = text


data_path = Path().cwd() / 'data'
data_path.mkdir(exist_ok=True)
with open(data_path / 'docs.json', 'w') as f:
    json.dump(docs, f)
