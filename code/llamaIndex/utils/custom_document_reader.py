import os
from tqdm import tqdm
from typing import List
from grobid_client.grobid_client import GrobidClient
import xml.etree.ElementTree as ET
from llama_index.core import Document

class CustomDocumentReader:
    def __init__(self, input_dir, cache_dir, config_path='./config.json'):
        # convert pdf to xml file
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.client = GrobidClient(config_path=config_path)

    def _convert_pdf_to_xml(self):
        self.client.process(
            "processFulltextDocument", 
            input_path=self.input_dir, 
            output=self.cache_dir,
            n=20,
            verbose=True
        )

    def _get_full_text(self, element):
        text = element.text or ''
        for subelement in element:
            text += ET.tostring(subelement, encoding='unicode', method='text')
            if subelement.tail:
                text += subelement.tail
        return text.strip()

    def _read_file(self, file_path):
        file_dict = {}
        tree = ET.parse(file_path)
        root = tree.getroot()

        namespace = {'tei': 'http://www.tei-c.org/ns/1.0'}

        # title
        title = root.find('.//tei:teiHeader/tei:fileDesc/tei:titleStmt/tei:title', namespaces=namespace)
        file_dict['title'] = title.text

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
        if abstract:
            file_dict['abstract'] = abstract.text

        # Body
        body = root.findall('.//tei:text/tei:body/tei:div', namespaces=namespace)

        for i, child in enumerate(body):
            ps = child.findall('.//tei:p', namespaces=namespace)
            if len(ps) == 0:
                continue
            head = child.find('.//tei:head', namespaces=namespace).text
            content = '\n'.join([self._get_full_text(p) for p in ps])
            file_dict[head] = content

        return file_dict

    def load_data(self) -> List[Document]:
        self._convert_pdf_to_xml()

        file_documents = []
        filenames = [filename for filename in os.listdir(self.cache_dir) if filename.endswith('tei.xml')]

        for filename in filenames:
            file_path = os.path.join(self.cache_dir, filename)
            file_dict = self._read_file(file_path)
            paper_content = "Title: {}\n\n".format(file_dict['title'])
            i = 1
            for k, v in file_dict.items():
                if k in ['title', 'authors']:
                    continue
                paper_content += f'[{i}. {k}]\n{v}\n\n'
                i += 1
            file_dict['file_name'] = filename

            file_document = Document(
                text=paper_content,
                metadata=file_dict
            )

            file_documents.append(file_document)
            os.remove(file_path)

        return file_documents

if __name__ == '__main__':
    root_path = '../../..'
    input_path = os.path.abspath(os.path.join(root_path, './data'))
    output_path = os.path.abspath(os.path.join(root_path, './code/llamaIndex/.cache'))
    CustomDocumentReader(input_path, output_path).load_data()
