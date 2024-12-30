import os
from tqdm import tqdm
import re
from llama_index.core import Document
from langdetect import detect
from tqdm import tqdm
from lxml import etree
from xml.etree.ElementTree import XMLPullParser
from typing import List
import html

class WikipediaDumpReader():
    def __init__(
        self,
        input_dir
    ):
        self.input_dir = input_dir
    
    def _clean_text(self, raw_text):
        cleaned_text = html.unescape(raw_text)
        cleaned_text = re.sub(r'\{\{.*?\}\}', '', raw_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"'''(.*?)'''", r'\1', cleaned_text)
        cleaned_text = re.sub(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]', r'\1', cleaned_text)
        return cleaned_text

    def _get_abstract(self, text, title, raw_text):
        # Split the text into lines
        abstract = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("{{") and line.endswith("}}"):
                continue
            abstract.append(line)
        abstract = self._clean_text("".join(abstract))
        if len(abstract) <= 0:
        #     print(raw_text[:10000])
        #     print(">>>>>>>>>>>>>>>")
        #     print(abstract.strip())
            print(f"Not find abstract at title {title}")
        # assert len(abstract) > 0, f"Not find abstract at title {title}"
            
        
        return abstract

    def _read_page(self, page):
        file_dict = {}
        file_dict['title'] = page['title']
        page_text = page['text']
        file_dict['sections'] = {}
        
        paper_content = "Title: {}\n\n".format(file_dict['title'])
        start = len(paper_content)
        
        raw_sections = re.split(r'(?m)^==\s*', page_text)
        old_section_title = None
        no_abstract = False
        for i, section in enumerate(raw_sections):
            if i == 0:
                # Process abstract
                abstract = self._get_abstract(section, page['title'], page['raw_page'])
                old_section_title = 'abstract'
                if len(abstract) > 0:
                    paper_content += f'{abstract}\n\n'
                    file_dict['sections']['abstract'] = [start, len(paper_content)-2]
                else:
                    no_abstract = True
            else:
                # Process body
                lines = section.split('\n')
                section_title = lines[0].strip('= ')
                section_content = self._clean_text("\n".join(lines[1:]))
                if not section.startswith('='):
                    assert old_section_title == 'abstract' or len(file_dict['sections'][old_section_title]) > 0,\
                        f"No section at title {page['title']} with section {old_section_title}\n{file_dict['sections'][old_section_title]}"
                    old_section_title = section_title
                    start = len(paper_content)
                    paper_content += f'{section_content}\n\n'
                    file_dict['sections'][section_title] = [start, len(paper_content)-2]
                else:
                    assert old_section_title is not None, 'old_section_title shouldn\'t be None'
                    paper_content += f'{section_content}\n\n'
                    file_dict['sections'][old_section_title] = [start, len(paper_content)-2]
        if len(file_dict['sections']) == 0:
            print(f"[documetn reader] Detect invalided document with no sections {page['title']}\n{page['raw_page']}")
            return None, None
        elif detect(paper_content):
            file_document = Document(
                text=paper_content,
                metadata=file_dict
            )
            return file_document, no_abstract
        
    def _go_over_string_element(self, xml_data):
        # Parse the string as a file-like object
        root = etree.fromstring(xml_data)
        redirect = root.find('redirect')
        title = root.findtext('title')
        text_element = root.find(".//text")  # Find the text element (can be nested)
        if redirect is not None or text_element is None or text_element.text is None: return None
        text = text_element.text.strip()
        
        return {'title': title, 'text': text, 'raw_page': xml_data}
        
    def _read_file(self, input_file_path):
        total_file_size = os.path.getsize(input_file_path)
        raw_page = None
        end_page = False
        parser = XMLPullParser(events=('start', 'end'))
        
        pages_key = {}
        documents = []
        
        with tqdm(total=total_file_size, desc="Processing", unit="B", unit_scale=True) as pbar:
            with open(input_file_path, 'r') as input_file:
                for line in input_file:
                    parser.feed(line)
                    for event, element in parser.read_events():
                        if event == 'start' and 'page' in element.tag:
                            raw_page = []
                        if event == 'end' and 'page' in element.tag:
                            end_page = True
                    # add line
                    if isinstance(raw_page, list):
                        raw_page.append(line)
                    if end_page:
                        raw_page = ''.join(raw_page)
                        page = self._go_over_string_element(raw_page)
                        if page:
                            for key in page:
                                pages_key[key] = pages_key.get(key, 0) + 1
                            document, no_abstract = self._read_page(page)
                            if document:
                                documents.append(document)
                                pages_key['no_abstract'] = pages_key.get('no_abstract', 0) + (1 if no_abstract else 0)
                            pbar.set_postfix_str(
                                f"{pages_key['no_abstract']}/{pages_key['title']}  {(pages_key['no_abstract']/pages_key['title'])*100:.2f}%"
                            )
                        raw_page = None
                        end_page = False
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
        file_documents = []
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.xml'):
                file_documents.extend(self._read_file(os.path.join(self.input_dir, filename)))
        unique_documents = self.remove_duplicate_documents(file_documents)
        # english_documents = self.remove_non_english_documents(unique_documents)
        return unique_documents

if __name__ == '__main__':
    root_path = '../../..'
    input_path = os.path.abspath(os.path.join(root_path, './data'))
    output_path = os.path.abspath(os.path.join(root_path, './code/llamaIndex/.cache'))
    WikipediaDumpReader(input_path, output_path).load_data()
    