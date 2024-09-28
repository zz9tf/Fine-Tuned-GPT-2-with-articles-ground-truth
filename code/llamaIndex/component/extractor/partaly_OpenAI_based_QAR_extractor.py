import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from tqdm import tqdm
from typing import List
from llama_index.core.schema import BaseNode
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import random
from component.io import save_nodes_jsonl, load_nodes_jsonl
from component.extractor.openai_QAR_extractor import OpenAIBasedQARExtractor

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

