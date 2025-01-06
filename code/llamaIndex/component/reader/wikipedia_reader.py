import os
from tqdm import tqdm
import re
import json
from llama_index.core import Document
from langdetect import detect
from tqdm import tqdm
from lxml import etree
from xml.etree.ElementTree import XMLPullParser
from typing import List
import html
from multiprocessing.pool import ThreadPool

class WikipediaDumpReader():
    def __init__(
        self,
        input_dir,
        cache_dir,
        worker: int=5,
        pages_per_batch: int=100
    ):
        self.input_dir = input_dir
        self.cache_dir = cache_dir
        self.worker = worker
        self.pages_per_batch = pages_per_batch
    
    def _write_to_disk(self, batch_id, batch):
        cache_file_path = os.path.join(self.cache_dir, f'raw_page_{batch_id}_not_finish.jsonl')
        with open(cache_file_path, 'w') as cache_file:
            for page in batch:
                cache_file.write(json.dumps({'page': page}) + '\n')
        os.rename(cache_file_path, os.path.join(self.cache_dir, f'raw_page_{batch_id}.jsonl'))
    
    def _read_file(self, input_file_path):
        total_file_size = os.path.getsize(input_file_path)
        parser = XMLPullParser(events=('start', 'end'))
        raw_page = None
        end_page = False
        page_num = 0
        batch_id = 0
        current_batch = []
        all_batches = []
        update_len = 0
        
        # Load all batches
        with open(input_file_path, 'r') as input_file:
            with tqdm(total=total_file_size, desc="Processing", unit="B", unit_scale=True) as pbar:
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
                        current_batch.append(raw_page)
                        if len(current_batch) > self.pages_per_batch:
                            all_batches.append([batch_id, current_batch])
                            batch_id += 1
                            current_batch = []
                        page_num += 1
                        raw_page = None
                        end_page = False
                    if update_len > 1e10:
                        pbar.set_postfix_str(f"page {page_num}")
                        pbar.update(update_len)
                        update_len = 0
                    else:
                        update_len += len(line)
                    if len(all_batches) >= 10:
                        # Write to disk using a ThreadPool for parallel processing of batches
                        with ThreadPool(self.worker) as pool, tqdm(total=len(all_batches), desc="process batches") as pbar:
                            # Wrap the pool.imap with tqdm to display a progress bar
                            for _ in pool.imap(lambda x: self._write_to_disk(x[0], x[1]), all_batches):
                                pbar.update(1)  # Update the progress bar for each processed batch
                        all_batches = []
        # If there are remaining pages in the last batch, add them
        if current_batch:
            all_batches.append([batch_id, current_batch])

        if len(all_batches) > 0:
            # Write to disk using a ThreadPool for parallel processing of batches
            with ThreadPool(self.worker) as pool, tqdm(total=len(all_batches), desc="process batches") as pbar:
                # Wrap the pool.imap with tqdm to display a progress bar
                for _ in pool.imap(lambda x: self._write_to_disk(x[0], x[1]), all_batches):
                    pbar.update(1)  # Update the progress bar for each processed batch
        
    def _parse_files(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.xml'):
                self._read_file(os.path.join(self.input_dir, filename))
    
    def _processed_xml(self):
        for filename in os.listdir(self.cache_dir):
            if "raw_page" in filename:
                return True
        return False
    
    def _clean_text(self, raw_text):
        cleaned_text = html.unescape(raw_text)
        cleaned_text = re.sub(r".*}}", "", raw_text, count=1, flags=re.DOTALL)
        cleaned_text = re.sub(r'\{\{.*?\}\}', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r"'''(.*?)'''", r'\1', cleaned_text)
        cleaned_text = re.sub(r'\[\[(?:[^\]|]+\|)?([^\]]+)\]\]', r'\1', cleaned_text)
        return cleaned_text

    def _get_abstract(self, text, title, raw_text):
        # Split the text into lines
        abstract = []
        print_str = ""
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
            print_str += f"Not find abstract at title {title}\n"
            
        # assert len(abstract) > 0, f"Not find abstract at title {title}"  
        
        return abstract, print_str

    def _read_page(self, page):
        if page['title'].lower().startswith('file:'):
            return None, None, ''
        file_dict = {}
        print_str = ""
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
                abstract, abstract_print_str = self._get_abstract(section, page['title'], page['raw_page'])
                print_str += abstract_print_str
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
        
        # Keys to remove
        keys_to_remove = ['references', 'external links', 'see also', 'further reading']

        # Remove keys safely
        for key in keys_to_remove:
            # Use a case-insensitive comparison to find the key
            matched_key = next((k for k in file_dict if k.lower() == key.lower()), None)
            if matched_key:
                file_dict.pop(matched_key)
        
        if len(file_dict['sections']) == 0:
            print_str += f"[documetn reader] Detect invalided document with no sections {page['title']}\n{page['raw_page']}\n"
            return None, None, print_str
        else:
            file_document = Document(
                text=paper_content,
                metadata=file_dict
            )
            return file_document, no_abstract, print_str
        
    def _go_over_string_element(self, xml_data):
        # Parse the string as a file-like object
        root = etree.fromstring(xml_data)
        redirect = root.find('redirect')
        title = root.findtext('title')
        text_element = root.find(".//text")  # Find the text element (can be nested)
        if redirect is not None or text_element is None or text_element.text is None: return None
        text = text_element.text.strip()
        
        return {'title': title, 'text': text, 'raw_page': xml_data}
        
    def _process_batch(self, batch_id, page_chunk):
        documents = []
        no_abstract_num = 0
        print_str = f"Start batch id: {batch_id}\n"

        for raw_page in page_chunk:
            page = self._go_over_string_element(raw_page)
            if page:
                document, no_abstract, page_print_str = self._read_page(page)
                print_str += page_print_str
                if document:
                    documents.append(document)
                    if no_abstract: no_abstract_num += 1
        return documents, no_abstract_num, print_str
    
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
    
    def load_data(self):
        if not self._processed_xml():
            self._parse_files()
        filenames = [filename for filename in os.listdir(self.cache_dir) if 'raw_page' in filename]
        filenames.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        
        all_batches = []
        documents = []
        no_abstract_num = 0
        
        with tqdm(total=len(filenames), desc="process file") as pbar:
            for filename in filenames:
                current_batch = []
                with open(os.path.join(self.cache_dir, filename), 'r') as cache_file:
                    for line in cache_file:
                        data = json.loads(line)
                        current_batch.append(data['page'])
                pbar.update(1)
                
                batch_id = int(filename.split('.')[0].split('_')[-1])
                all_batches.append([batch_id, current_batch])
                if len(all_batches) >= 10:
                    pbar.set_postfix_str(
                        f"extracting documents {no_abstract_num}/{len(documents)}  {(no_abstract_num/len(documents) if len(documents) != 0 else 0)*100:.2f}%"
                    )
                    with ThreadPool(self.worker) as pool:
                        for batch_documents, batch_no_abstract_num, print_str in pool.imap(
                            lambda x: self._process_batch(x[0], x[1]), all_batches
                        ):
                            # print(print_str)
                            documents.extend(batch_documents)
                            no_abstract_num += batch_no_abstract_num
                    
                    pbar.set_postfix_str(
                        f"{no_abstract_num}/{len(documents)}  {(no_abstract_num/len(documents))*100:.2f}%"
                    )
                    
                    all_batches = []
        
        if current_batch:
            all_batches.append([batch_id, current_batch])
            
        if len(all_batches) > 0:
            pbar.set_postfix_str(
                f"extracting documents {no_abstract_num}/{len(documents)}  {(no_abstract_num/len(documents))*100:.2f}%"
            )
            with ThreadPool(self.worker) as pool:
                for batch_documents, batch_no_abstract_num, print_str in pool.imap(
                    lambda x: self._process_batch(x[0], x[1]), all_batches
                ):
                    print(print_str)
                    documents.extend(batch_documents)
                    no_abstract_num += batch_no_abstract_num
                
            pbar.set_postfix_str(
                f"{batch_no_abstract_num}/{len(documents)}  {(batch_no_abstract_num/len(documents))*100:.2f}%"
            )
                
        return documents

if __name__ == '__main__':
    root_path = '../../..'
    input_path = os.path.abspath(os.path.join(root_path, './data'))
    output_path = os.path.abspath(os.path.join(root_path, './code/llamaIndex/.cache'))
    WikipediaDumpReader(input_path, output_path).load_data()
    