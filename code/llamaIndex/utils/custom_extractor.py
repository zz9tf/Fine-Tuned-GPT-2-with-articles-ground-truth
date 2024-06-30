from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import infer_auto_device_map, init_empty_weights
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.llms.ollama import Ollama
from tqdm import tqdm
from openai import OpenAI

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
class QAExtractor():
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

        node.metadata["questions_this_excerpt_can_answer_and_corresponding_answers"] = str(generated_text).strip()
        if "questions_this_excerpt_can_answer_and_corresponding_answers" not in node.excluded_llm_metadata_keys:
            node.excluded_llm_metadata_keys.append("questions_this_excerpt_can_answer_and_corresponding_answers")

    
    def extract(self, nodes: List[BaseNode]):
        for node in tqdm(nodes):
            self._extract_metadata_from_node(node)


class OpenAIBasedExtractor():
    def __init__(self):
        self.client = OpenAI()
    
        batch_input_file = self.client.files.create(
        file=open("batchinput.jsonl", "rb"),
        purpose="batch"
        )

        batch_input_file_id = batch_input_file.id

        self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            "description": "nightly eval job"
            }
        )
        
        content = self.client.files.content("file-xyz123")