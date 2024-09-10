import os
from typing import List
import shutil
from langdetect import detect
from tqdm import tqdm
from grobid_client.grobid_client import GrobidClient
import xml.etree.ElementTree as ET
from llama_index.core import Document
from langdetect import detect

class CustomDocumentReader:
    def __init__(
            self, 
            input_dir, 
            cache_dir, 
            config_path=None, 
            remove_cache=True
    ) -> None:
        # convert pdf to xml file
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.json') if config_path is None else config_path
        self.remove_cache = remove_cache

    def _convert_pdf_to_xml(self):
        def process_pdf_flies(copied_files_num, tmp_folder_path, client):
            if copied_files_num % 1000 == 0:
                client.process(
                    "processFulltextDocument", 
                    input_path=tmp_folder_path, 
                    output=self.cache_dir,
                    n=20,
                    verbose=True
                )
                if os.path.exists(tmp_folder_path):
                    shutil.rmtree(tmp_folder_path, ignore_errors=True)
                os.makedirs(tmp_folder_path, exist_ok=True)
        
        client = GrobidClient(config_path=self.config_path)

        ext = '.grobid.tei.xml'
        xml_files = set([file[:-len(ext)] for file in os.listdir(self.cache_dir) if file.endswith(ext)])

        copied_files_num = 0
        tmp_folder_path = os.path.join(self.input_dir, 'tmp')
        for file in tqdm(os.listdir(self.input_dir), desc='file'):
            file_path = os.path.join(self.input_dir, file)
            if file.split('.')[0] in xml_files or os.path.isdir(file_path): continue
            process_pdf_flies(copied_files_num, tmp_folder_path, client)
            shutil.copy(file_path, tmp_folder_path)         
            copied_files_num += 1

        process_pdf_flies(copied_files_num, tmp_folder_path, client)

    def _get_full_text(self, element):
        text = element.text or ''
        for subelement in element:
            text += ET.tostring(subelement, encoding='unicode', method='text')
            if subelement.tail:
                text += subelement.tail
        return text.strip()

    def _read_file(self, file_path, filename):
        file_dict = {}
        tree = ET.parse(file_path)
        root = tree.getroot()

        namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # title
        title = root.find('.//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title', namespaces=namespace)
        if title.text is not None:
            file_dict['title'] = title.text if len(title.text) < 150 else title.text[:150]
        else:
            file_dict['title'] = filename if len(filename) < 150 else filename[:150]
        # authors
        authors = root.findall('.//tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author/tei:persName', namespaces=namespace)
        file_dict['authors'] = []
        for author in authors:
            forename = author.findall('tei:forename', namespaces=namespace)
            if len(forename) == 0:
                continue
            forename = ' '.join(name.text for name in forename)
            surname = ' '.join([name.text for name in author.findall('tei:surname', namespaces=namespace)])
            author_name = forename + ' ' + surname
            file_dict['authors'].append(author_name)
        
        # abstract
        abstract = root.find('.//tei:teiHeader/tei:profileDesc/tei:abstract/tei:div/tei:p', namespaces=namespace)
        if abstract is not None:
            file_dict['abstract'] = abstract.text

        # Body
        body = root.findall('.//tei:text/tei:body/tei:div', namespaces=namespace)

        caption = None
        for i, child in enumerate(body):
            ps = child.findall('.//tei:p', namespaces=namespace)
            if len(ps) == 0:
                continue
            head = child.find('.//tei:head', namespaces=namespace)
            content = '\n'.join([self._get_full_text(p) for p in ps])
            try:
                if not hasattr(head, 'text') and content is not None:
                    file_dict[caption] += content
                else:
                    if head.text.lower() == 'abstract':
                        caption = 'abstract'
                    else:
                        caption = head.text
                    file_dict[caption] = content
            except Exception as e:
                continue

        return file_dict

    def _load_data(self):
        file_documents = []
        filenames = [filename for filename in os.listdir(self.cache_dir) if filename.endswith('tei.xml')]

        for filename in filenames:
            file_path = os.path.join(self.cache_dir, filename)
            file_dict = self._read_file(file_path, filename)
            file_dict['sections'] = {}
            paper_content = "Title: {}\n\n".format(file_dict['title'])
            i = 1
            copy_dict = file_dict.copy()
            for title, section in copy_dict.items():
                if title in ['title', 'authors', 'sections']:
                    continue
                start = len(paper_content)
                paper_content += f'{section}\n\n'
                file_dict['sections'][title] = [start, len(paper_content)-2]
                del file_dict[title]
                i += 1
            file_dict['file_name'] = filename.replace('grobid.tei.xml', 'pdf')
            if len(file_dict['sections']) == 0:
                print(f"[documetn reader] Detect invalided document with no sections {file_dict['file_name']}")
            elif detect(paper_content):
                file_document = Document(
                    text=paper_content,
                    metadata=file_dict
                )

                file_documents.append(file_document)
            if self.remove_cache:
                os.remove(file_path)

        return file_documents

    def remove_duplicate_documents(self, documents):
        unique_document = []
        file_head_set = set()
        for document in documents:
            head = document.text[:800]
            if head not in file_head_set:
                file_head_set.add(head)
                unique_document.append(document)
        return unique_document

    def remove_non_english_documents(self, documents):
        english_documents = []
        for document in documents:
            head = document.text[:800]
            try:
                if detect(head) == 'en':
                    english_documents.append(document)
            except Exception as e:
                print(e)
        return english_documents

    def load_data(self) -> List[Document]:
        self._convert_pdf_to_xml()
        file_documents = self._load_data()
        unique_documents = self.remove_duplicate_documents(file_documents)
        english_documents = self.remove_non_english_documents(unique_documents)
        return english_documents

if __name__ == '__main__':
    root_path = '../../..'
    input_path = os.path.abspath(os.path.join(root_path, './data'))
    output_path = os.path.abspath(os.path.join(root_path, './code/llamaIndex/.cache'))
    CustomDocumentReader(input_path, output_path).load_data()
