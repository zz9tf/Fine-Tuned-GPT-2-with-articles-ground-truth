import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import csv
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from llama_index.core.schema import BaseNode
from llama_index.core.llms.llm import LLM
from tqdm import tqdm
from component.schema import TemplateSchema, QAR
from component.extractor.pydantic import CustomPydanticOutputParser
from component.extractor.utils import parse_obj_to_str

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