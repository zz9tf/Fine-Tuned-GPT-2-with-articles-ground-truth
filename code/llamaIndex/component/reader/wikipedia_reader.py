import os
from tqdm import tqdm
import re
from llama_index.core import Document
from langdetect import detect
from tqdm import tqdm
from lxml import etree
from typing import List

class WikipediaDumpReader():
    def __init__(
        self,
        input_file_path
    ):
        self.input_file_path = input_file_path
    
    def _clean_text(self, raw_text):
        print(raw_text)
        cleaned_text = re.sub(r'\{\{.*?\}\}', '', raw_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"'''(.*?)'''", r'\1', cleaned_text)
        cleaned_text = re.sub(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]', r'\1', cleaned_text)
        return cleaned_text

    def _read_page(self, page):
        file_dict = {}
        file_dict['title'] = page['title']
        page_text = page['text']
        file_dict['sections'] = {}
        paper_content = "Title: {}\n\n".format(file_dict['title'])
        
        # Load abstract
        abstract_pattern = r"\n'''(.*?)'''(.*?)(?=\n\=|$)"
        abstract_match = re.search(abstract_pattern, page_text, re.DOTALL)
        abstract = self._clean_text(abstract_match.group(0).strip()) if abstract_match else None
        assert abstract_match is not None, f'Not find abstract at title {file_dict['title']}'
        start = len(paper_content)
        paper_content += f'{abstract}\n\n'
        file_dict['sections']['abstract'] = [start, len(paper_content)-2]
        
        # Load body
        raw_sections = re.split(r'(?m)^==\s*', page_text)
        old_section_title = None
        for section in raw_sections[1:]:
            lines = section.split('\n')
            section_title = lines[0].strip('= ')
            section_content = self._clean_text("\n".join(lines[1:]))
            if not section.startswith('='):
                old_section_title = section_title
                start = len(paper_content)
                paper_content += f'{section_content}\n\n'
                file_dict['sections'][section_title] = [start, len(paper_content)-2]
            else:
                assert old_section_title is not None, 'old_section_title shouldn\'t be None'
                paper_content += f'{section_content}\n\n'
                file_dict['sections'][old_section_title] = [start, len(paper_content)-2]
        if len(file_dict['sections']) == 0:
            print(f"[documetn reader] Detect invalided document with no sections {file_dict['file_name']}")
        elif detect(paper_content):
            file_document = Document(
                text=paper_content,
                metadata=file_dict
            )
            return file_document

    def _read_file(self, input_file_path):
        total_file_size = os.path.getsize(input_file_path)

        page = None
        documents = []
        with open(input_file_path, 'rb') as file, tqdm(total=total_file_size, desc="Processing", unit="B", unit_scale=True) as pbar:
            for event, element in etree.iterparse(file, events=('start', 'end')):
                pbar.update(file.tell() - pbar.n)
                if event == 'start' and 'page' in element.tag:
                    page = {}
                if page:
                    key = element.tag.split('}')[-1]
                    if key in ['title', 'text']:
                        page[key] = element.text
                    if key == 'redirect':
                        page[key] = element.attrib['title']
                    # else:
                    #     page[key] = element
                if event == 'end' and 'page' in element.tag:
                    if 'redirect' not in page:
                        document = self._read_page(page)
                        if document:
                            documents.append(document)
                    page = None
        return documents
    
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
        file_documents = self._read_file(self.input_file_path)
        unique_documents = self.remove_duplicate_documents(file_documents)
        english_documents = self.remove_non_english_documents(unique_documents)
        return english_documents

if __name__ == '__main__':
    root_path = '../../..'
    input_path = os.path.abspath(os.path.join(root_path, './data'))
    output_path = os.path.abspath(os.path.join(root_path, './code/llamaIndex/.cache'))
    WikipediaDumpReader(input_path, output_path).load_data()
    