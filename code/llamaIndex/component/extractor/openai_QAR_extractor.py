import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import csv
import json
from tqdm import tqdm
import time
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from openai import OpenAI
from llama_index.llms.openai import OpenAI as llama_index_openai
import langchain_core
from llama_index.core.schema import BaseNode
from component.schema import TemplateSchema, LCTemp
from component.extractor.utils import parse_obj_to_str

class OpenAIBasedQARExtractor(ABC):
    def __init__(
        self,
        model_name: str,
        cache_dir: str = '.',
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
        node.metadata['additional_to_be_embedding_keys'] = [self._prompt_metadata_key]
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
        entry = {
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
        print(entry['body']['messages'])
        exit()
        # Langchain end
        return entry 

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
        return response

    def _update_nodes(
        self, nodes: List[BaseNode], finished_batches: dict, node_dict: dict):
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
                    node.metadata['additional_to_be_embedding_keys'] = [self._prompt_metadata_key]
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
            cache_dir: Optional[str] = None,
            csv_file_name: Optional[str] = None
        ):
        csv_file_name = csv_file_name if csv_file_name is not None else f"{index_id}-{action}-QAR.csv"
        cache_dir = self.cache_dir if cache_dir is None else cache_dir
        csv_file = open(os.path.join(cache_dir, csv_file_name), 'w', newline='', encoding='utf-8')
        self.dataset_writer = csv.DictWriter(csv_file, fieldnames=['node_id', 'qar_num', 'input_text', 'raw_response', 'objs'])
        self.dataset_writer.writeheader()

        # target_nodes = self._get_target_nodes(nodes)
        target_nodes = nodes
        # TODO remove this
        # target_nodes = nodes[:2]
        # target_nodes[0].metadata['requirements'] = {'qar_num': 2}

        if self.mode == 'immediately':
            for node in tqdm(target_nodes):
                self._extract_metadata_from_node_immediately(node)
        elif self.mode == 'batch':
            self._extract_metadata_from_nodes_batch(target_nodes)
        
        csv_file.close()

