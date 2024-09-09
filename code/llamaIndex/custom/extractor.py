##########################################################################
# Extractor
import os
from custom.llm import get_llm

def get_extractors(self, extractor_config):
    llm_config = self.prefix_config['llm'][extractor_config['llm']]
    
    if extractor_config['type'] == 'OpenAIBasedExtractor':
        return OpenAIBasedQARExtractor(
            model_name=extractor_config['llm'],
            cache_dir=os.path.abspath(os.path.join(self.root_path, self.config['cache'])),
            mode=extractor_config['mode'],
            embedding_only=extractor_config.get('embedding_only', True)
        )
    elif extractor_config['type'] in ['OllamaBasedExtractor', 'HuggingfaceBasedExtractor']:
        return CustomLLMBasedQARExtractor(
            llm_self=get_llm(self, llm_config),
            embedding_config=self.prefix_config['embedding_model'][extractor_config['embedding_model']],
            embedding_only=extractor_config.get('embedding_only', True)
        )
    elif extractor_config['type'] == PartalyOpenAIBasedQARExtractor:
        return PartalyOpenAIBasedQARExtractor(
            model_name=extractor_config['llm'],
            cache_dir=os.path.abspath(os.path.join(self.root_path, self.config['cache'])),
            mode=extractor_config['mode'],
            embedding_only=extractor_config.get('embedding_only', True)
        )
##########################################################################

from typing import Dict, List, Optional, Callable
from abc import ABC, abstractmethod
import os
import csv
import json
import time
from datetime import datetime
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.llms.llm import LLM
from tqdm import tqdm
from openai import OpenAI
from llama_index.llms.openai import OpenAI as llama_index_openai
from custom.schema import TemplateSchema
from custom.schema import QAR
from custom.schema import LCTemp
from custom.pydantic import CustomPydanticOutputParser

def parse_obj_to_str(objs):
    if objs is not None:
        return json.dumps([obj.dict() if obj is not None else {} for obj in objs])
    return str([])

class CustomLLMBasedQARExtractor(ABC):
    def __init__(
        self,
        llm: LLM,
        prompt_template: dict = TemplateSchema.prompt_template_ollama,
        embedding_only: bool = True
    ) -> None:
        """Init params."""
        self._model = llm
        self._prompt_metadata_key, self._prompt_template = prompt_template
        self.embedding_only = embedding_only
        self.pydantic_parser = CustomPydanticOutputParser(output_cls=QAR)

    def _extract_metadata_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract metadata from a node and return it's metadata dict."""
        requirements = {}
        if 'requirements' in node.metadata:
            requirements = node.metadata['requirements']
            del node.metadata['requirements']
        context_str = self._prompt_template.format(context_str=node.text, qar_num=requirements.get('qar_num', 5))
        input_text = self.pydantic_parser.format(query=context_str)
        generated_text = self._model.complete(input_text).text.strip()

        outputs = self.pydantic_parser.parse(generated_text)
        for output in outputs:
            output_dict = output.dict()
            output_dict['node_id'] = node.id_
            self.dataset_writer.writerow(output_dict)

        node.metadata[self._prompt_metadata_key] = parse_obj_to_str(outputs)

        if self._prompt_metadata_key not in node.excluded_llm_metadata_keys and self.embedding_only:
            node.excluded_llm_metadata_keys.append(self._prompt_metadata_key)

    @abstractmethod
    def _get_target_nodes(self, nodes: List[BaseNode]):
        return nodes

    def extract(
            self, 
            nodes: List[BaseNode], 
            index_id: Optional[str] = 'index_id',
            action: Optional[str] = 'action',
            cache_path: Optional[str] = ''
        ):
        csv_file = open(os.path.join(cache_path, f"{index_id}-{action}-QAR.csv"), 'w', newline='')
        self.dataset_writer = csv.DictWriter(csv_file, fieldnames=['node_id', 'Question', 'Answer', 'Reason'])
        self.dataset_writer.writeheader()
        target_nodes = self._get_target_nodes(nodes)
        
        for node in tqdm(target_nodes):
            self._extract_metadata_from_node(node)
        csv_file.close()

import langchain_core

class OpenAIBasedQARExtractor(ABC):
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        mode: str = 'immediately',
        system_prompt: str = TemplateSchema.system_prompt,
        prompt_metadata_key: str = TemplateSchema.prompt_metadata_key_openai,
        # prompt_template: str = TemplateSchema.prompt_template_openai,
        prompt_template: langchain_core.prompts.prompt.PromptTemplate = LCTemp.prompt_template,
        embedding_only: bool = True
    ) -> None:
        self._model = llama_index_openai(model=model_name, api_key=os.environ.get('OPENAI_API_KEY'))
        self._client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.mode = mode
        self.cache_dir = cache_dir
        self._system_prompt = system_prompt
        self._prompt_metadata_key = prompt_metadata_key
        self._prompt_template = prompt_template
        self.embedding_only = embedding_only
        # self.pydantic_parser = CustomPydanticOutputParser(output_cls=QAR)
        self.pydantic_parser = LCTemp.parser

    def _extract_metadata_from_node_immediately(self, node: BaseNode) -> Dict[str, str]:
        """Extract metadata from a node and return it's metadata dict."""
        requirements = {}
        if 'requirements' in node.metadata:
            requirements = node.metadata['requirements']
        # llama_index custom start
        # context_str = self._prompt_template.format(context_str=node.text, qar_num=requirements.get('qar_num', 5))
        # input_text = self.pydantic_parser.format(query=context_str)
        # llama_index custom end
        # Langchain start
        input_text = self._prompt_template.format(context_str=node.text, qar_num=requirements.get('qar_num', 5))
        # Langchain end
        response = self._model.complete(input_text).text.strip()
        parsed_objs = self.pydantic_parser.parse(response)
        # llama_index custom
        # objs_str = parse_obj_to_str(parsed_objs)
        # Langchain
        objs_str = parse_obj_to_str(parsed_objs.qars)
        output_dict = {}
        output_dict['node_id'] = node.id_
        output_dict['qar_num'] = requirements['qar_num']
        output_dict['input_text'] = input_text
        output_dict['raw_response'] = response
        output_dict['objs'] = objs_str
        self.dataset_writer.writerow(output_dict)
        node.metadata[self._prompt_metadata_key] = objs_str
        if self._prompt_metadata_key not in node.excluded_llm_metadata_keys and self.embedding_only:
            node.excluded_llm_metadata_keys.append(self._prompt_metadata_key)

        if 'requirements' in node.metadata:
            del node.metadata['requirements']

    def _create_a_batch(self, now, input_file_path):
        with open(input_file_path, "rb") as input_file:
            batch_input_file = self._client.files.create(
                file=input_file,
                purpose="batch"
            )

        batch_input_file_id = batch_input_file.id

        batch = self._client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        batch_info_path = os.path.join(self.cache_dir, f"{now}---batchinput-response.json")
        with open(batch_info_path, 'w') as file:
            json.dump(batch.to_dict(), file, indent=4)

        return batch, batch_info_path

    def _generate_an_entry(self, node):
        requirements = {'qar_num': 5}
        if 'requirements' in node.metadata:
            requirements = node.metadata['requirements']
        else:
            node.metadata['requirements'] = requirements
        # llama_index custom start
        # context_str = self._prompt_template.format(context_str=node.text, qar_num=requirements.get('qar_num', 5))
        # context_str = self.pydantic_parser.format(query=context_str)
        # llama_index custom end
        # Langchain start
        context_str = self._prompt_template.format(context_str=node.text, qar_num=requirements.get('qar_num', 5))
        # Langchain end
        return {
            "custom_id": node.id_,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": context_str}
                    ],
                "max_tokens": 4096
            }
        }

    def _create_batches_from_nodes(self, nodes, request_num=45000):
        node_id = 0
        batches = []
        node_dict = {} # {node.id_: {'node': node, 'input_text': input_text}}
        input_file_paths = {} # {now : input_file_path}
        batch_info_paths = {} # {id : {"path": batch_info_path, 'now': now}}

        total_batches = int(len(nodes)/request_num) + (len(nodes) % request_num > 0)

        with tqdm(total=total_batches, desc="Creating batches", unit="batch") as pbar:
            for node in tqdm(nodes):
                node_dict[node.id_] = {'node': node}
                if node_id % request_num == 0:
                    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    input_file_path = os.path.join(self.cache_dir, f"{now}---batchinput.jsonl")
                    input_file_paths[now] = input_file_path
                    file = open(input_file_path, 'w')

                entry = self._generate_an_entry(node)
                node_dict[node.id_]['input_text'] = entry['body']['messages'][1]['content']
                json_line = json.dumps(entry)
                file.write(json_line + "\n")
                if (node_id+1) % request_num == 0:
                    file.close()
                    batch, batch_info_path = self._create_a_batch(now, input_file_path)
                    batch_info_paths[batch.id] = {'path': batch_info_path, 'now': now}
                    batches.append(batch)
                    pbar.n = len(batches)
                    pbar.refresh()
                node_id += 1

            if (node_id+1) % request_num != 0:
                file.close()
                batch, batch_info_path = self._create_a_batch(now, input_file_path)
                batch_info_paths[batch.id] = {'path': batch_info_path, 'now': now}
                batches.append(batch)
                pbar.n = len(batches)
                pbar.refresh()
        
        return batches, node_dict, input_file_paths, batch_info_paths

    def _check_batches_results(self, uncompleted_batches, input_file_paths, batch_info_paths, finished_batches, node_dict):
        uncompleted_batches_copy = uncompleted_batches.copy()
        has_failed = False
        for _, batch in uncompleted_batches.items():
            # Refresh the batch response
            batch = self._client.batches.retrieve(batch.id)
            if batch.status in ['completed', 'expired', 'failed', 'cancelled']:
                del uncompleted_batches_copy[batch.id]
                # If the batch is completed
                if batch.status == 'completed':
                    if batch.request_counts.completed == batch.request_counts.total:
                        # Remove previous cache files
                        batch_info_path = batch_info_paths[batch.id]
                        os.remove(batch_info_path['path'])
                        now = batch_info_path['now']
                        input_file_path = input_file_paths[now]
                        os.remove(input_file_path)

                        # Save results to cache
                        batch_content = self._client.files.content(batch.output_file_id)
                        batch_content =[json.loads(response) for response in batch_content.text.strip().split('\n')]
                        for response_dict in batch_content:
                            response_dict['input_text'] = node_dict[response_dict['custom_id']]['input_text']
                        output_file_path = os.path.join(self.cache_dir, f'{now}---batchoutput.json')
                        with open(output_file_path, 'w') as output_file:
                            json.dump(batch_content, output_file, indent=4)
                        finished_batches[output_file_path] = batch_content

                    else:
                        print(f"Error complete with failed {batch.request_counts.failed} at batch {batch.id}")
                        output_file_path = os.path.join(self.cache_dir, f'{now}--Failed--{batch.id}--batchoutput.json')
                        output_file = open(output_file_path, 'w')
                        output_file.close()
                        has_failed = True
                if batch.status in ['expired', 'failed', 'cancelled']:
                    print(f"Get {batch.status} at batch {batch.id}")
                    has_failed = True
        return uncompleted_batches_copy, finished_batches, has_failed

    def _processing_batches(self, uncompleted_batches, input_file_paths, batch_info_paths, node_dict):
        finished_batches = {}
        has_failed = False
        total_batches = len(uncompleted_batches)
        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
            while len(uncompleted_batches) > 0:
                uncompleted_batches, finished_batches, has_failed = self._check_batches_results(uncompleted_batches, input_file_paths, batch_info_paths, finished_batches, node_dict)
                pbar.n = total_batches - len(uncompleted_batches)
                pbar.refresh()
                # Check status each 5 minutes
                time.sleep(5)
        return has_failed, finished_batches
    
    def _format_response(self, response):
        if response.startswith('```json'):
            response = response[7:-3].strip()
        else:
            response = response
        return response

    def _update_nodes(self, nodes, finished_batches, node_dict):
        total_nodes = len(nodes)
        with tqdm(total=total_nodes, desc="Updating nodes", unit="nodes") as pbar:
            for output_file_path, batch_content in finished_batches.items():
                for request in batch_content:
                    # get response
                    response = request['response']['body']['choices'][0]['message']['content'].strip()
                    node = node_dict[request['custom_id']]['node']
                    # get parsed_objs
                    # get objs_str

                    # llama_index custom start
                    # objs_str = parse_obj_to_str(parsed_objs)
                    # llama_index custom end

                    # Langchain start
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print(node_dict[node.id_]['input_text'])
                    print("response:")
                    response = self._format_response(response)
                    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    try:
                        parsed_objs = self.pydantic_parser.parse(response)
                    except Exception as e:
                        print(f"Exception when parse response: {e}")
                        parsed_objs = None
                    if parsed_objs is not None:
                        parsed_objs = parsed_objs.qars
                    objs_str = parse_obj_to_str(parsed_objs)
                    # Langchain end
                    # write to csv
                    output_dict = {}
                    output_dict['node_id'] = node.id_
                    print(node.metadata)
                    qar_num = node.metadata['requirements']['qar_num']
                    output_dict['qar_num'] = qar_num
                    output_dict['input_text'] = node_dict[node.id_]['input_text']
                    output_dict['raw_response'] = response
                    output_dict['objs'] = objs_str
                    self.dataset_writer.writerow(output_dict)
                    node.metadata[self._prompt_metadata_key] = objs_str
                    if self._prompt_metadata_key not in node.excluded_llm_metadata_keys and self.embedding_only:
                        node.excluded_llm_metadata_keys.append(self._prompt_metadata_key)
                    
                    del node.metadata['requirements']
                    # pbar.n is updated nodes number, which should add 1 each time
                    pbar.n += 1
                    pbar.refresh()
                os.remove(output_file_path)

    def _extract_metadata_from_nodes_batch(self, nodes: List[BaseNode]) -> Dict[str, str]:
        batches, node_dict, input_file_paths, batch_info_paths = self._create_batches_from_nodes(nodes)
        uncompleted_batches = {batch.id: batch for batch in batches}
        has_failed, finished_batches = self._processing_batches(uncompleted_batches, input_file_paths, batch_info_paths, node_dict)
        
        if has_failed:
            print("has failed")

        self._update_nodes(nodes, finished_batches, node_dict)

    @abstractmethod
    def _get_target_nodes(self, nodes: List[BaseNode]):
        return nodes

    def extract(
            self, 
            nodes: List[BaseNode], 
            index_id: Optional[str] = 'index_id',
            action: Optional[str] = 'action',
            cache_path: Optional[str] = '',
            csv_file_name: Optional[str] = None
        ):
        csv_file_name = csv_file_name if csv_file_name is not None else f"{index_id}-{action}-QAR.csv"
        cache_path = self.cache_dir if cache_path is None else cache_path
        csv_file = open(os.path.join(cache_path, csv_file_name), 'w', newline='')
        self.dataset_writer = csv.DictWriter(csv_file, fieldnames=['node_id', 'qar_num', 'input_text', 'raw_response', 'objs'])
        self.dataset_writer.writeheader()

        target_nodes = self._get_target_nodes(nodes)
        # TODO remove this
        # target_nodes = nodes[:2]
        # target_nodes[0].metadata['requirements'] = {'qar_num': 2}

        if self.mode == 'immediately':
            for node in tqdm(target_nodes):
                self._extract_metadata_from_node_immediately(node)
        elif self.mode == 'batch':
            self._extract_metadata_from_nodes_batch(target_nodes)
        
        csv_file.close()


from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random
from custom.io import save_nodes_jsonl, load_nodes_jsonl
class PartalyOpenAIBasedQARExtractor(OpenAIBasedQARExtractor):

    def get_score(self, text):
        tokenized_input = self.tokenizer(text,truncation=True, padding=True, return_tensors='pt')
        logits = self.sentence_model(**tokenized_input).logits
        probabilities = logits.softmax(dim=1)
        return probabilities.detach().numpy()[0][1]

    def organize_nodes(self, target_nodes: List[BaseNode], nodes: list[BaseNode]):
        # Initialize
        document_id2section_nodes = {}
        id_2_node = {}
        document_id2paragraph_nodes = {}
        document_id2multi_sentence_nodes = {}
        for node in nodes:
            requirements = {'qar_num': 1}
            level = node.metadata['level']
            id_2_node[node.id_] = node
            if level == 'document':
                requirements['qar_num'] = 3
                target_nodes.append(node)
            node.metadata['requirements'] = requirements

        for node in nodes:
            level = node.metadata['level']
            if level == 'section':
                document_id = node.ref_doc_id
                if document_id not in document_id2section_nodes:
                    document_id2section_nodes[document_id] = []
                document_id2section_nodes[document_id].append(node)
            elif level == 'paragraph':
                document_id = id_2_node[node.ref_doc_id].ref_doc_id
                if document_id not in document_id2paragraph_nodes:
                    document_id2paragraph_nodes[document_id] = []
                document_id2paragraph_nodes[document_id].append(node)
            elif level == 'multi-sentences':
                document_id = id_2_node[id_2_node[node.ref_doc_id].ref_doc_id].ref_doc_id
                if document_id not in document_id2multi_sentence_nodes:
                    document_id2multi_sentence_nodes[document_id] = []
                document_id2multi_sentence_nodes[document_id].append(node)
        return document_id2section_nodes, document_id2paragraph_nodes, document_id2multi_sentence_nodes

    def _classify_nodes(self, nodes):
        """Classify nodes into selected and non-selected based on key words."""
        key_words = ['introduction', 'discussion', 'conclusion', 'result', 'method', 'materials', 'analyses']
        selected_nodes = []
        non_selected_nodes = []
        
        for node in nodes:
            if any(key_word in node.metadata.get('section_title', '') for key_word in key_words):
                selected_nodes.append(node)
                if len(selected_nodes) >= 7:
                    break
            else:
                non_selected_nodes.append(node)
        
        return selected_nodes, non_selected_nodes

    def _fill_nodes(self, selected_nodes, non_selected_nodes, target_count=7):
        """Ensure that the selected nodes list has the target number of nodes."""
        additional_nodes_needed = target_count - len(selected_nodes)
        
        if additional_nodes_needed > 0:
            if len(non_selected_nodes) <= additional_nodes_needed:
                selected_nodes.extend(non_selected_nodes)
                return selected_nodes, additional_nodes_needed - len(non_selected_nodes)
            else:
                selected_nodes.extend(random.sample(non_selected_nodes, additional_nodes_needed))
        
        return selected_nodes, 0

    def _get_target_section_nodes(
            self, 
            target_nodes, 
            document_id2section_nodes
        ):
        document_id2paragraph_select_nums = {}
        
        for document_id, nodes in tqdm(document_id2section_nodes.items(), desc="Getting target section nodes..."):
            document_id2paragraph_select_nums[document_id] = 5
            selected_nodes, non_selected_nodes = self._classify_nodes(nodes)
            selected_nodes, new_additional_num = self._fill_nodes(selected_nodes, non_selected_nodes)
            document_id2paragraph_select_nums[document_id] += new_additional_num
            target_nodes.extend(selected_nodes)
        return document_id2paragraph_select_nums

    def _get_target_paragraph_nodes(self, target_nodes, document_id2paragraph_nodes, document_id2paragraph_select_nums):
        for document_id, nodes in tqdm(document_id2paragraph_nodes.items(), desc="Getting target paragraph nodes..."):
            score2nodes = {}
            for node in nodes:
                text = node.get_content()
                if len(text) < 800:
                    score = self.get_score(text)
                    if score not in score2nodes:
                        score2nodes[score] = []
                    score2nodes[score].append(node)
            sorted_scores = sorted(score2nodes.keys(), reverse=True)

            selected_nodes = []
            select_num = document_id2paragraph_select_nums[document_id]
            for score in sorted_scores:
                if len(selected_nodes) >= select_num:
                    break
                nodes_at_score = score2nodes[score]
                selected_nodes.extend(nodes_at_score)
            selected_nodes = selected_nodes[:select_num]

            target_nodes.extend(selected_nodes)
        
    def _get_target_multi_sentence_nodes(self, target_nodes, document_id2multi_sentence_nodes, select_num=5):
        for _, nodes in tqdm(document_id2multi_sentence_nodes.items(), desc="Getting target multi-sentence nodes..."):
            score2nodes = {}
            for node in nodes:
                text = node.get_content()
                if len(text) < 800:
                    score = self.get_score(text)
                    if score not in score2nodes:
                        score2nodes[score] = []
                    score2nodes[score].append(node)
            sorted_scores = sorted(score2nodes.keys(), reverse=True)

            selected_nodes = []
            for score in sorted_scores:
                if len(selected_nodes) >= select_num:
                    break
                nodes_at_score = score2nodes[score]
                selected_nodes.extend(nodes_at_score)
            selected_nodes = selected_nodes[:select_num]

            target_nodes.extend(selected_nodes)
    
    def _get_target_nodes(self, nodes: List[BaseNode]):
        target_nodes = []
        target_file_path = os.path.join(self.cache_dir, "target_nodes.jsonl")

        # Check if the target file exists
        if os.path.exists(target_file_path):
            # Load target nodes
            target_nodes = load_nodes_jsonl(file_path=target_file_path)
            target_node_id2target_node = {}
            for node in target_nodes:
                target_node_id2target_node[node.id_] = node
            # Overwrite selected nodes with target nodes
            for i in range(len(nodes)):
                if nodes[i].id_ in target_node_id2target_node:
                    node[i] = target_node_id2target_node[node.id_]
        else:
            document_id2section_nodes, document_id2paragraph_nodes, document_id2multi_sentence_nodes = self.organize_nodes(target_nodes, nodes)
            document_id2paragraph_select_nums = self._get_target_section_nodes(target_nodes, document_id2section_nodes)

            # prepare models
            self.sentence_model = AutoModelForSequenceClassification.from_pretrained("MomochiKyaru/glyco-paper-sentence",token=os.getenv('GLYCO_TOKEN'))
            self.paragraph_model = AutoModelForSequenceClassification.from_pretrained("MomochiKyaru/glyco-paper-paragraph", token=os.getenv('GLYCO_TOKEN'))
            self.tokenizer =AutoTokenizer.from_pretrained('MomochiKyaru/glyco-paper-sentence',token=os.getenv('GLYCO_TOKEN'))

            self._get_target_paragraph_nodes(target_nodes, document_id2paragraph_nodes, document_id2paragraph_select_nums)
            self._get_target_multi_sentence_nodes(target_nodes, document_id2multi_sentence_nodes)
            save_nodes_jsonl(target_file_path, nodes=target_nodes)
        
        return target_nodes

