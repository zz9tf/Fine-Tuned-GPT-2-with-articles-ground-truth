from typing import Dict, List
import os
import torch
import json
import time
from datetime import datetime
from llama_index.core.storage.docstore import SimpleDocumentStore
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.llms.ollama import Ollama
from tqdm import tqdm
from openai import OpenAI
from llama_index.llms.openai import OpenAI as llama_index_openai

DEFAULT_QUESTION_GEN_TMPL="""\
Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} questions this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.

"""
# [TODO] Need to accelarate the model
class HuggingfaceBasedExtractor():
    def __init__(
        self,
        model_name,
        no_split_modules: str = None,
        cache_dir: str = None,
        num_questions: int = 5,
        prompt_template: str = DEFAULT_QUESTION_GEN_TMPL,
        embedding_only: bool = True
    ) -> None:
        """Init params."""
        if num_questions < 1:
            raise ValueError("questions must be >= 1")
        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        max_memory = {i: '20000MB' for i in range(torch.cuda.device_count())}
        max_memory[0] = '18000MB'
        max_memory['cpu'] = '120GiB'
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes = [no_split_modules])
        print(device_map)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map=device_map, offload_folder="/workspace/.cache")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_questions = num_questions
        self._prompt_template = prompt_template

    def _extract_metadata_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract metadata from a node and return it's metadata dict."""

        context_str = node.get_content(metadata_mode=MetadataMode.ALL)
        input_text = self._prompt_template.format(context_str=context_str, num_questions=self.num_questions)
        inputs = self._tokenizer(input_text, return_tensors='pt').to(self._device)
        outputs = self._model.generate(
            inputs.input_ids,
            max_length=2048,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        # generated_text = self._model.complete(input_text)

        return {"questions_this_excerpt_can_answer_and_corresponding_answers_": str(generated_text).strip()}
    
    def extract(self, nodes):
        for node in tqdm(nodes):
            metadata = self._extract_metadata_from_node(node)
            for k, v in metadata.items():
                node.metadata[k] = v

prompt_template_ollama = """\
"Here is the context:
{context_str}

Here is the format of question, answer, and reason(QAR) template:
----------------------------------------------------------------------------------
<Pair number, representing which QAR you are at, like 1, 2, 3>
Question:<Question content, you should place a specific question which is unlikely to be found elsewhere and is unique comparing with other questions>

Answer:<Answer content, you should place a specific answer combining with the offered context>

Reason:<Reason content, you should explain why this question and answer are unlikely to be found elsewhere and are unique comparing with each other>
----------------------------------------------------------------------------------

Following by this template, given the contextual information, generate 5 QAR.\
Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
"""

class OllamaBasedExtractor():
    def __init__(
        self,
        model_name: str,
        prompt_template: str = prompt_template_ollama,
        embedding_only: bool = True
    ) -> None:
        """Init params."""
        self._model = Ollama(model=model_name, request_timeout=60.0)
        self._prompt_template = prompt_template
        self.embedding_only = embedding_only

    def _extract_metadata_from_node(self, node: BaseNode) -> Dict[str, str]:
        """Extract metadata from a node and return it's metadata dict."""

        context_str = node.get_content(metadata_mode=MetadataMode.ALL)
        input_text = self._prompt_template.format(context_str=context_str)
        generated_text = self._model.complete(input_text)

        node.metadata["questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons"] = str(generated_text).strip()
        if "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons" not in node.excluded_llm_metadata_keys and self.embedding_only:
            node.excluded_llm_metadata_keys.append("questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons")

    
    def extract(self, nodes: List[BaseNode]):
        for node in tqdm(nodes):
            self._extract_metadata_from_node(node)

system_prompt = """\
You are a highly knowledgeable reasearch assistant tasked with generating insightful questions, detailed answers, and \
thorough reasoning based on the provided parts of papers.\
"""

prompt_template_openai="""\
Here is the context:
{context_str}

Using this context, generate 5 specific questions that this context can uniquely answer. Ensure that these questions:
1. Are directly related to the provided context.
2. Highlight unique information or insights from the context.
3. Cannot be easily answered by general knowledge.
----------------------------------------------------------------------------------
Pair Number of Question, such as 1, 2, or 3.
Question:<Question content, you should place a specific question which is unlikely to be found elsewhere and is unique comparing with other questions>

Answer:<Answer content, you should place a specific answer combining with the offered context>

Reason:<Reason content, you should explain why this question and answer are unlikely to be found elsewhere and are unique comparing with each other>
----------------------------------------------------------------------------------
Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer.
"""

class OpenAIBasedExtractor():
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        mode: str = 'immediately',
        system_prompt: str = system_prompt,
        prompt_template: str = prompt_template_openai,
        embedding_only: bool = True
    ) -> None:
        self._model = llama_index_openai(model=model_name, api_key=os.environ.get('OPENAI_API_KEY'))
        self._client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.mode = mode
        self.cache_dir = cache_dir
        self._prompt_template = prompt_template
        self.embedding_only = embedding_only

    def _extract_metadata_from_node_immediately(self, node: BaseNode) -> Dict[str, str]:
        """Extract metadata from a node and return it's metadata dict."""
        context_str = node.get_content(metadata_mode=MetadataMode.ALL)
        input_text = self._prompt_template.format(context_str=context_str)
        generated_text = self._model.complete(input_text)

        node.metadata["questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons"] = str(generated_text).strip()
        if "questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons" not in node.excluded_llm_metadata_keys and self.embedding_only:
            node.excluded_llm_metadata_keys.append("questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons")

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
        return {
            "custom_id": node.id_,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._prompt_template.format(context_str=node.text)}
                    ],
                "max_tokens": 4096
            }
        }

    def _create_batches_from_nodes(self, nodes, request_num=45000):
        node_id = 0
        batches = []
        node_dict = {}
        input_file_paths = {} # {now : input_file_path}
        batch_info_paths = {} # {id : {"path": batch_info_path, 'now': now}}

        total_batches = int(len(nodes)/request_num) + (len(nodes) % request_num > 0)

        with tqdm(total=total_batches, desc="Creating batches", unit="batch") as pbar:
            for node in tqdm(nodes):
                node_dict[node.id_] = node
                if node_id % request_num == 0:
                    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    input_file_path = os.path.join(self.cache_dir, f"{now}---batchinput.jsonl")
                    input_file_paths[now] = input_file_path
                    file = open(input_file_path, 'w')

                json_line = json.dumps(self._generate_an_entry(node))
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

    def _check_batches_results(self, uncompleted_batches, input_file_paths, batch_info_paths, finished_batches):
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
                        output_file_path = os.path.join(self.cache_dir, f'{now}---batchoutput.json')
                        with open(output_file_path, 'w') as output_file:
                            json.dump(batch_content, output_file, indent=4)
                        finished_batches[output_file_path] = batch_content

                    else:
                        print(f"Error complete with failed {batch.request_counts.failed} at batch {batch.id}")
                        has_failed = True
                if batch.status in ['expired', 'failed', 'cancelled']:
                    print(f"Get {batch.status} at batch {batch.id}")
                    has_failed = True
        return uncompleted_batches_copy, finished_batches, has_failed

    def _processing_batches(self, uncompleted_batches, input_file_paths, batch_info_paths):
        finished_batches = {}
        has_failed = False
        total_batches = len(uncompleted_batches)
        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
            while len(uncompleted_batches) > 0:
                uncompleted_batches, finished_batches, has_failed = self._check_batches_results(uncompleted_batches, input_file_paths, batch_info_paths, finished_batches)
                pbar.n = total_batches - len(uncompleted_batches)
                pbar.refresh()
                # Check status each 5 minutes
                time.sleep(5)
        return has_failed, finished_batches

    def _extract_metadata_from_nodes_batch(self, nodes: List[BaseNode]) -> Dict[str, str]:
        batches, node_dict, input_file_paths, batch_info_paths = self._create_batches_from_nodes(nodes)
        uncompleted_batches = {batch.id: batch for batch in batches}
        has_failed, finished_batches = self._processing_batches(uncompleted_batches, input_file_paths, batch_info_paths)
        
        if has_failed:
            exit()
        
        total_nodes = len(nodes)
        with tqdm(total=total_nodes, desc="Updating nodes", unit="nodes") as pbar:
            for output_file_path, batch_content in finished_batches.items():
                for request in batch_content:
                    content = request['response']['body']['choices'][0]['message']['content']
                    node_dict[request['custom_id']].metadata["questions_this_excerpt_can_answer_and_corresponding_answers_and_reasons"] = content
                    # pbar.n is updated nodes number, which should add 1 each time
                    pbar.n += 1
                    pbar.refresh()
                os.remove(output_file_path)
            
    def extract(self, nodes: List[BaseNode]):
        if self.mode == 'immediately':
            for node in tqdm(nodes):
                self._extract_metadata_from_node_immediately(node)
        elif self.mode == 'batch':
            self._extract_metadata_from_nodes_batch(nodes)
        

        