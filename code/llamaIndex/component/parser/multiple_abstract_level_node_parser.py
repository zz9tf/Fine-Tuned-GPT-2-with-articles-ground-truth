import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import io
import uuid
import json
from typing import Any, Dict, List, Optional, Sequence
from tqdm import tqdm
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import BaseNode, Document, NodeRelationship, MetadataMode, TextNode
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from component.schema import TemplateSchema
from component.parser.response_synthesis import TreeSummarize
from component.parser.semantic import SemanticSplitter
from llama_index.core.node_parser import SentenceSplitter

class MultipleAbstractLevelNodeParser():
    """Multiple abstract level node parser.

    Splits a document into a multiple abstract hierarchy structure using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list
    """
    def __init__(
        self,
        _cache_process_path: str,
        _cache_process_file: str,
        _sentences_splitter: SentenceSplitter,
        _tree_summarizer: TreeSummarize,
        _semantic_splitter: SemanticSplitter
    ):
        self._cache_process_path = _cache_process_path
        self._cache_process_file = _cache_process_file
        self._chunk_levels = ['document', 'section', 'paragraph', 'multi-sentences']
        self._sentences_splitter = _sentences_splitter
        self._tree_summarizer = _tree_summarizer
        self._semantic_splitter = _semantic_splitter

    @classmethod
    def from_defaults(
        cls,
        llm_config: Dict,
        embedding_config: Dict,
        cache_dir_path: str = os.path.abspath('.'),
        cache_file_name: str = 'CustomHierarchicalNodeParser_cache.jsonl'
    ):
        """get a MultipleAbstractLevelNodeParser"""
        cache_process_file = None
        cache_process_path = os.path.join(cache_dir_path, cache_file_name)

        sentences_splitter = SentenceSplitter(
            chunk_size=128,
            chunk_overlap=20,
            include_metadata=False,
            include_prev_next_rel=False
        )

        tree_summarizer = TreeSummarize.from_defaults(
            query_str=TemplateSchema.tree_summary_section_q_Tmpl,
            summary_str=TemplateSchema.tree_summary_summary_Tmpl,
            qa_prompt=TemplateSchema.tree_summary_qa_Tmpl,
            llm_config=llm_config
        )
        
        semantic_splitter = SemanticSplitter(buffer_size=1, breakpoint_percentile_threshold=97, embed_model_config=embedding_config)

        return cls(
            _cache_process_path = cache_process_path,
            _cache_process_file = cache_process_file,
            _sentences_splitter = sentences_splitter,
            _tree_summarizer = tree_summarizer,
            _semantic_splitter = semantic_splitter
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomHierarchicalNodeParser"

    def save_nodes(self, nodes: list[BaseNode]) -> None:
        for node in nodes:
            json.dump(node.to_dict(), self._cache_process_file)
            self._cache_process_file.write('\n')
            self._cache_process_file.flush()

    def _text_split(self, section) -> List[str]:
        paragraphs = []
        pre_splitter = SentenceSplitter(
            chunk_size=500, 
            chunk_overlap=20, 
            include_metadata=False, 
            include_prev_next_rel=False
        )
        force_splitter = SentenceSplitter(
            chunk_size=400, 
            chunk_overlap=20, 
            include_metadata=False, 
            include_prev_next_rel=False
        )
        for p in section.split('\n'):
            if len(p.strip()) == 0:
                continue
            if len(p) > 800:
                preprocessed_ps = pre_splitter.split_text_metadata_aware(p, '')
                for preprocessed_p in preprocessed_ps:
                    ps = self._semantic_splitter.parse_text(preprocessed_p)
                    paragraphs.extend(ps)
                    # final_ps = []
                    # for p_new in ps:
                    #     final_new_ps = force_splitter.split_text_metadata_aware(p_new, '')
                    #     final_ps.extend([p for p in final_new_ps if len(p.strip()) > 0])
                    # paragraphs.extend(final_ps)
            else:
                paragraphs.append(p)
        return paragraphs

    def _add_line_breaks(
            self,
            document_node
        ):
        text = ""
        sections = document_node.metadata['sections']
        new_sections = {}
        for title, (start, end) in sections.items():
            section_text = document_node.get_content()[start: end]
            # split section into suitable size with splitter \n
            section_text = '\n'.join(self._text_split(section_text))
            start = len(text)
            text += section_text + '\n\n'
            new_sections[title] = (start, len(text)-2)
        document_node.text = text
        document_node.metadata['sections'] = new_sections

    def _preprocess_documents(self, documents: Sequence[Document]):
        self._semantic_splitter.load_embed()
        for document in tqdm(documents, desc="preprocessing documents..."):
            document.metadata['original_content'] = document.text
            self._add_line_breaks(document_node=document)
            document.metadata['level'] = 'preprocessed_document'
            self.save_nodes([document])
        self._semantic_splitter.del_embed()

    def _update_exclude_keys(self, node):
        node.excluded_embed_metadata_keys = list(node.metadata.keys())
        node.excluded_llm_metadata_keys = list(node.metadata.keys())

    def _get_document_node_from_document(
        self,
        document: Document,
    ) -> List[BaseNode]:

        relationships = {NodeRelationship.SOURCE: document.as_related_node_info()}

        document_node = TextNode(
            id_=str(uuid.uuid4()),
            text=document.text,
            embedding=document.embedding,
            excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
            metadata_seperator=document.metadata_seperator,
            metadata_template=document.metadata_template,
            text_template=document.text_template,
            relationships=relationships,
        )

        # Update document_node attributions
        document_node.metadata.update(document.metadata)
        document_node.metadata['level'] = 'document'
        # self._add_start_char_idx_and_end_char_idx(document_node, document)
        self._update_exclude_keys(document_node)
            
        return document_node
    
    def _summary_content(self, texts):
        new_texts = [text for text in texts if len(text.strip()) > 0]
        # if len(texts) == 0:
        #     print(texts)
        #     print(new_texts)
        #     input()
        summary, _ = self._tree_summarizer.generate_response_hs(
            texts=new_texts,
            num_children=20
        )
        return summary
        # return ""

    def _summary_sections(
        self,
        document: BaseNode,
        document_node: BaseNode
    ):
        titles = []
        sections = []
        summaries = []
        # Get summary of section contents
        abstract_paragraphs = None
        for title, (start, end) in document.metadata['sections'].items():
            section = document_node.get_content()[start: end]
            if title == 'abstract':
                print(f"> abstract ({start},{end}) <")
                abstract_paragraphs = section.split('\n')
            else:
                titles.append(title)
                sections.append(section)
                print(f"> {title} section ({start},{end}) <")
                summary = self._summary_content(section.split('\n'))
                summaries.append(summary)
        
        # Get summary of abstract
        summary_for_document = ''
        if abstract_paragraphs is not None:
            print("> abstract abstract_paragraphs <")
            summary_for_document = self._summary_content(abstract_paragraphs)
        else:
            print("> abstract summaries <")
            summary_for_document = self._summary_content(summaries)
        
        result_dict = {
            "summary_for_document": summary_for_document,
            "summaries": summaries,
            "titles": titles,
            "sections": sections
        }

        return result_dict

    def _add_previous_and_next_relationship(self, i: int, nodes: Sequence[BaseNode], node: BaseNode):
        if (
            i > 0
            and node.source_node
            and nodes[i - 1].source_node
            and nodes[i - 1].source_node.node_id == node.source_node.node_id
        ):
            node.relationships[NodeRelationship.PREVIOUS] = {
                'node_id': [nodes[i - 1].node_id]
            }
        if (
            i < len(nodes) - 1
            and node.source_node
            and nodes[i + 1].source_node
            and nodes[i + 1].source_node.node_id == node.source_node.node_id
        ):
            node.relationships[NodeRelationship.NEXT] = {
                'node_id': [nodes[i - 1].node_id]
            }

    def _add_parent_child_relationship(self, child: BaseNode, parent: BaseNode) -> None:
        """Add parent/child relationship between nodes."""
        child_list = parent.relationships.get(NodeRelationship.CHILD, {'node_id': []})
        child_list['node_id'].append(child.node_id)

        parent.relationships[NodeRelationship.CHILD] = child_list
        child.relationships[NodeRelationship.PARENT] = {
            'node_id': [parent.node_id]
        }

    def _get_section_nodes_from_document_node(
        self,
        document: BaseNode,
        document_node: BaseNode
    ) -> List[BaseNode]:
        result_dict = self._summary_sections(document, document_node)

        # Update document node
        document_node.text = result_dict["summary_for_document"]

        # Get all section nodes
        all_nodes: List[BaseNode] = build_nodes_from_splits(
            result_dict["summaries"], document_node, id_func=self.id_func
        )

        # Update section nodes attributions
        for i, node in enumerate(all_nodes):
            title = result_dict['titles'][i]
            section = result_dict['sections'][i]        
            # Update metadata
            node.metadata['section_title'] = title
            node.metadata['original_content'] = section
            node.metadata['level'] = 'section'
            # self._add_start_char_idx_and_end_char_idx(node, document_node)
            # Add relationships
            self._add_previous_and_next_relationship(i, all_nodes, node)
            self._add_parent_child_relationship(node, document_node)
            self._update_exclude_keys(node)

        return all_nodes

    def get_paragraphs_from_text(self, content):
        """remove all lines smaller than 50"""
        result = []
        buffer = ""

        for line in content.split('\n'):
            if len(buffer) > 128:
                result.append(buffer)
            else:
                buffer += line
        # Append any remaining buffer content
        if len(buffer) > 0:
            result.append(buffer)
        return result

    def _get_paragraph_nodes_from_section_node(
        self,
        section_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        original_content = section_node.metadata['original_content']
        splits = self.get_paragraphs_from_text(original_content)
        del section_node.metadata['original_content']
        all_nodes.extend(
            build_nodes_from_splits(splits, section_node)
        )
        for i, node in enumerate(all_nodes):
            # Update metadata
            node.metadata['level'] = 'paragraph'
            # self._add_start_char_idx_and_end_char_idx(node, section_node)
            # Update relationship
            self._add_previous_and_next_relationship(i, all_nodes, node)
            self._add_parent_child_relationship(node, section_node)
            self._update_exclude_keys(node)

        return all_nodes

    def _get_multiple_sentences_nodes_from_paragraph_node(
        self,
        paragraph_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes = self._sentences_splitter._parse_nodes([paragraph_node])
        all_nodes = [node for node in all_nodes if len(node.get_content().strip()) > 10]
        for i, node in enumerate(all_nodes):
            # Update metadata
            node.metadata['level'] = 'multi-sentences'
            # self._add_start_char_idx_and_end_char_idx(node, paragraph_node)
            # Update relationship
            self._add_previous_and_next_relationship(i, all_nodes, node)
            self._add_parent_child_relationship(node, paragraph_node)
            self._update_exclude_keys(node)

        return all_nodes

    def _get_nodes_from_one_document(
        self,
        document: Document,
        pbar=None
    ):
        if pbar is not None:
            pbar.set_description("parsing documents - getting a document node")
            pbar.set_postfix(small_step="0/1")
        document_node = self._get_document_node_from_document(document=document)
        if pbar is not None:
            pbar.set_postfix(small_step="1/1")
            pbar.update(0.25)

        if pbar is not None:
            pbar.set_description("parsing document nodes - getting section nodes")
            pbar.set_postfix(small_step="0/1")
        section_nodes = self._get_section_nodes_from_document_node(document, document_node)
        if pbar is not None:
            pbar.set_postfix(small_step="1/1")
            pbar.update(0.25)

        if pbar is not None:
            pbar.set_description("parsing section nodes - getting paragraph nodes")
        paragraph_nodes = []
        for i, section_node in enumerate(section_nodes):
            if pbar is not None:
                pbar.set_postfix(small_step=f"{i}/{len(section_nodes)}")
            paragraph_nodes.extend(self._get_paragraph_nodes_from_section_node(section_node))
            if pbar is not None:
                pbar.set_postfix(small_step=f"{i+1}/{len(section_nodes)}")
        if pbar is not None:
            pbar.update(0.25)

        if pbar is not None:
            pbar.set_description("parsing paragraph nodes - getting multi-sentences nodes")
        multi_sentences_nodes = []
        for i, paragraph_node in enumerate(paragraph_nodes):
            if pbar is not None:
                pbar.set_postfix(small_step=f"{i}/{len(paragraph_nodes)}")
            multi_sentences_nodes.extend(
                self._get_multiple_sentences_nodes_from_paragraph_node(paragraph_node)
            )
            if pbar is not None:
                pbar.set_postfix(small_step=f"{i+1}/{len(paragraph_nodes)}")
        if pbar is not None:
            pbar.update(0.25)
        
        # save nodes to cache
        self.save_nodes([document_node] + section_nodes + paragraph_nodes + multi_sentences_nodes)

    def _load_cache_nodes(self):
        if not os.path.exists(self._cache_process_path):
            return
        # Load the cache file
        file_size = os.path.getsize(self._cache_process_path)
        with open(self._cache_process_path, 'r') as cache_file:
            with tqdm(total=file_size, desc='Loading cache...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in cache_file:
                    node_dict = json.loads(line)
                    node = TextNode.from_dict(node_dict)
                    self._level2nodes[node.metadata['level']].append(node)
                    # Update progress bar based on bytes read
                    pbar.update(len(line))

    def _init_get_nodes_from_documents(self, documents):
        # init attributions
        self._level2nodes = {level:[] for level in self._chunk_levels + ['preprocessed_document']}
        # loading nodes
        self._load_cache_nodes()

        # Open cache file
        self._cache_process_file = open(self._cache_process_path, 'a+')
        # return nonfinished documents
        if len(self._level2nodes['document']) == 0:
            preprocessed_document_ids = {document.id_ for document in self._level2nodes['preprocessed_document']}
            preprocessed_documents = []
            nonpreprocess_documents = []
            for document in documents:
                if document.id_ in preprocessed_document_ids:
                    preprocessed_documents.append(document)
                else:
                    nonpreprocess_documents.append(document)
            if len(nonpreprocess_documents) > 0:
                self._preprocess_documents(nonpreprocess_documents)
            self._level2nodes = None
            self._tree_summarizer.load_llm()
            return preprocessed_documents + nonpreprocess_documents
        else:
            finished_document_ids = {document.ref_doc_id for document in self._level2nodes['document']}
            nonfinished_documents = [document for document in documents if document.id_ not in finished_document_ids]
            self._level2nodes = None
            self._tree_summarizer.load_llm()
            return nonfinished_documents

    def load_results(self):
        """load results saved in the cache file"""
        assert os.path.exists(self._cache_process_path), f"cache file {self._cache_process_path} doesn't exist."
        # Load the cache file
        all_nodes = []
        file_size = os.path.getsize(self._cache_process_path)
        with open(self._cache_process_path, 'r') as cache_file:
            with tqdm(total=file_size, desc='Reading cache...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                for line in cache_file:
                    try:
                        node_dict = json.loads(line)
                        node = TextNode.from_dict(node_dict)
                        if node.metadata['level'] != 'preprocessed_document':
                            all_nodes.append(node)
                    except Exception as e:
                        print(e)
                    # Update progress bar based on bytes read
                    pbar.update(len(line))
        return all_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = True,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress bar

        """
        non_finished_documents = self._init_get_nodes_from_documents(documents)
        print(f"not finished document: {len(non_finished_documents)}")
        
        if show_progress:
            with tqdm(total=len(documents), desc="parsing documents") as pbar:
                pbar.update(len(documents) - len(non_finished_documents))
                for _, document in enumerate(non_finished_documents):
                    self._get_nodes_from_one_document(document, pbar)
        else:
            for document in non_finished_documents:
                self._get_nodes_from_one_document(document)
        self._cache_process_file.close()
        return self.load_results()