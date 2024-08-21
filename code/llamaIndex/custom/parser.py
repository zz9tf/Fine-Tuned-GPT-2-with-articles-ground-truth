##########################################################################
# parser
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)

def get_parser(self, config, **kwargs):
    VALID_PARSER = self.prefix_config['parser'].keys()
    if config['type'] == 'SentenceSplitter':
        return SentenceSplitter(
            chunk_size=config.get('chunk_size', 1024), 
            chunk_overlap=config.get('chunk_overlap', 200)
        )
    elif config['type'] == 'SimpleFileNodeParser':
        return SimpleFileNodeParser()
    elif config['type'] == 'HierarchicalNodeParser':
        return HierarchicalNodeParser.from_defaults(
            chunk_sizes=config.get('chunk_size', [2048, 512, 128])
        )
    elif config['type'] == 'CustomHierarchicalNodeParser':
        return CustomHierarchicalNodeParser.from_defaults(
            llm_self=self,
            llm_config=self.prefix_config['llm'][config['llm']],
            embedding_config=self.prefix_config['embedding_model'][config['embedding_model']]
        )
    elif config['type'] == "ManuallyHierarchicalNodeParser":
        return ManuallyParser(
            cache_path=kwargs["cache_path"],
            cache_name=f'{kwargs["index_id"]}_{kwargs["step_id"]}_{kwargs["step_type"]}_{kwargs["action"]}',
            delete_cache=False
        )
    else:
        raise Exception("Invalid parser config. Please provide parser types {}".format(VALID_PARSER))
##########################################################################

import os
import io
import gc
import torch
import json
from tqdm import tqdm
from llama_index.core.embeddings import BaseEmbedding
from custom.io import load_nodes_jsonl, save_nodes_jsonl
from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import BaseNode, Document, NodeRelationship, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.llms import LLM
from llama_index.core.node_parser.relational.hierarchical import _add_parent_child_relationship
from custom.schema import TemplateSchema
from custom.response_synthesis import TreeSummarize
from custom.embedding import get_embedding_model
from custom.semantic import SemanticSplitter

class CustomHierarchicalNodeParser(NodeParser):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).
    """
    _cache_process_path: str = PrivateAttr()
    _cache_process_file: io.IOBase = PrivateAttr()
    _level2nodes: List[TextNode] = PrivateAttr()

    # The chunk level to use when splitting documents: document, section, paragraph, multi-sentences
    _chunk_levels: List[str] = PrivateAttr()

    _doc_id_to_document: Dict[str, Document] = PrivateAttr()

    _sentences_splitter: SentenceSplitter = PrivateAttr()
    
    _embedding_config: Dict = PrivateAttr()

    _llm_self = PrivateAttr()

    _llm_config: Dict = PrivateAttr()
    
    _tree_summarizer: TreeSummarize = PrivateAttr()

    @classmethod
    def from_defaults(
        cls,
        llm_self,
        llm_config: LLM,
        embedding_config: BaseEmbedding,
        cache_dir_path: str = None,
        cache_file_name: str = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None
    ) -> "CustomHierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        cls._chunk_levels = ["document", "section", "paragraph", "multi-sentences"]

        cls._cache_process_path = None
        cls._cache_process_file = None
        cls._level2nodes = {}
        if cache_dir_path is not None:
            cls._cache_process_path = os.path.join(cache_dir_path, cache_file_name)
        
        cls._embedding_config = embedding_config

        cls._sentences_splitter = SentenceSplitter(
            chunk_size=128, 
            chunk_overlap=20, 
            include_metadata=False, 
            include_prev_next_rel=False
        )

        cls._llm_self = llm_self
        cls._llm_config = llm_config
        cls._tree_summarizer = TreeSummarize.from_defaults(
            query_str=TemplateSchema.tree_summary_section_q_Tmpl,
            summary_str=TemplateSchema.tree_summary_summary_Tmpl,
            qa_prompt=TemplateSchema.tree_summary_qa_Tmpl,
            llm_self=llm_self,
            llm_config=llm_config
        )

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomHierarchicalNodeParser"

    def save_nodes(self, nodes: list[BaseNode]) -> None:
        if self._cache_process_path == None:
            return
        
        for node in nodes:
            json.dump(node.to_dict(), self._cache_process_file)
            self._cache_process_file.write('\n')
            self._cache_process_file.flush()

    def _get_document_node_from_document(
        self,
        document: Document,
    ) -> List[BaseNode]:
        document_content = document.get_content(metadata_mode=MetadataMode.NONE)

        # build node from document
        all_nodes = build_nodes_from_splits(
            [document_content],
            document,
            id_func=self.id_func,
        )
            
        return all_nodes

    def _summary_content(self, texts):
        print('summary')
        summary, _ = self._tree_summarizer.generate_response_hs(
            texts=texts,
            num_children=len(texts)
        )
        return summary

    def _text_split(self, section, document_node):
        paragraphs = []
        semantic_splitter = None
        for p in section.split('\n'):
            print('text split')
            if len(p) > 9000:
                print('split')
                self._tree_summarizer.del_llm()
                if semantic_splitter is None:
                    # semantic_splitter = SemanticSplitterNodeParser(buffer_size=2, breakpoint_percentile_threshold=95, embed_model=get_embedding_model(self._embedding_config))
                    semantic_splitter = SemanticSplitter(buffer_size=2, breakpoint_percentile_threshold=95, embed_model=get_embedding_model(self._embedding_config))
                # Get the total memory allocated on the GPU
                allocated_memory = torch.cuda.memory_allocated()
                # Get the total memory cached on the GPU
                cached_memory = torch.cuda.memory_reserved()

                # Convert bytes to megabytes (MB)
                allocated_memory_mb = allocated_memory / (1024 ** 3)
                cached_memory_mb = cached_memory / (1024 ** 3)

                print(f"Allocated Memory: {allocated_memory_mb:.2f} GB")
                print(f"Cached Memory: {cached_memory_mb:.2f} GB")
                node = build_nodes_from_splits([p], document_node, id_func=self.id_func)[0]
                paragraphs += [n.get_content() for n in semantic_splitter.get_nodes_from_documents([node])]
        if semantic_splitter is not None:
            del semantic_splitter
            torch.cuda.empty_cache()
            gc.collect()
            # Convert bytes to megabytes (MB)
            allocated_memory_mb = allocated_memory / (1024 ** 3)
            cached_memory_mb = cached_memory / (1024 ** 3)

            print(f"Allocated Memory: {allocated_memory_mb:.2f} GB")
            print(f"Cached Memory: {cached_memory_mb:.2f} GB")
            self._tree_summarizer.load_llm()
            # Convert bytes to megabytes (MB)
            allocated_memory_mb = allocated_memory / (1024 ** 3)
            cached_memory_mb = cached_memory / (1024 ** 3)

            print(f"Allocated Memory: {allocated_memory_mb:.2f} GB")
            print(f"Cached Memory: {cached_memory_mb:.2f} GB")
        return section.split('\n')

    def _get_section_nodes_from_document_node(
        self, 
        document_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        parent_document = self._doc_id_to_document.get(document_node.ref_doc_id, None)
        if parent_document is None:
            raise Exception(f"Parent document is not found with id {document_node.ref_doc_id}")

        titles = []
        sections = []
        summaries = []
        # Get summary of section contents
        abstract_paragraphs = None
        for title, (start, end) in parent_document.metadata['sections'].items():
            section = document_node.get_content()[start: end]
            if title == 'abstract':
                abstract_paragraphs = self._text_split(section, document_node)
            else:
                titles.append(title)
                sections.append(section)
                # summary = ''
                summary = self._summary_content(self._text_split(section, document_node))
                summaries.append(summary)
        
        # Get summary of abstract
        summary_for_document = ''
        if abstract_paragraphs != None:
            summary_for_document = self._summary_content(abstract_paragraphs)
        else:
            summary_for_document = self._summary_content(summaries)

        # Update document node
        origin_document_text = document_node.text
        document_node.metadata['original_content'] = origin_document_text
        document_node.text = summary_for_document

        # Update section nodes
        all_nodes.extend(
            build_nodes_from_splits(summaries, document_node, id_func=self.id_func)
        )

        for title, section, node in zip(titles, sections, all_nodes):
            node.metadata['section_title'] = title
            node.metadata['original_content'] = section

        return all_nodes

    def _get_paragraph_nodes_from_section_node(
        self,
        section_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        
        splits = section_node.metadata['original_content'].split('\n')

        for i, text in enumerate(splits):
            if i == 0:
                continue
            splits[i] = f"One Paragraph of {splits[0]}: {text}"
        all_nodes.extend(
            build_nodes_from_splits(splits, section_node, id_func=self.id_func)
        )

        return all_nodes

    def _get_multiple_sentences_nodes_from_paragraph_node(
        self,
        paragraph_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes = self._sentences_splitter._parse_nodes([paragraph_node])

        return all_nodes

    def _postprocess_parsed_nodes(
        self, 
        nodes: List[BaseNode], 
        chunk_level: str
    ) -> List[BaseNode]:
        for i, node in enumerate(nodes):
            parent_doc = self._doc_id_to_document.get(node.ref_doc_id, None)
            if parent_doc is not None:
                start_char_idx = parent_doc.text.find(
                        node.get_content(metadata_mode=MetadataMode.NONE)
                    )

                # update start/end char idx
                if start_char_idx >= 0:
                    node.start_char_idx = start_char_idx
                    node.end_char_idx = start_char_idx + len(
                        node.get_content(metadata_mode=MetadataMode.NONE)
                    )

                metadata = {k: v for k, v in parent_doc.metadata.items() if k not in ['sections']}
                metadata['level'] = chunk_level
                node.metadata.update(metadata)

                exclude_keys = list(metadata.keys())
                exclude_keys.remove('title')
                node.excluded_embed_metadata_keys.extend(exclude_keys)
                node.excluded_llm_metadata_keys.extend(exclude_keys)

                self._doc_id_to_document[node.id_] = parent_doc
                    
            if chunk_level != 'document':
                # establish prev/next relationships if nodes share the same source_node
                if (
                    i > 0
                    and node.source_node
                    and nodes[i - 1].source_node
                    and nodes[i - 1].source_node.node_id == node.source_node.node_id
                ):
                    node.relationships[NodeRelationship.PREVIOUS] = nodes[
                        i - 1
                    ].as_related_node_info()
                if (
                    i < len(nodes) - 1
                    and node.source_node
                    and nodes[i + 1].source_node
                    and nodes[i + 1].source_node.node_id == node.source_node.node_id
                ):
                    node.relationships[NodeRelationship.NEXT] = nodes[
                        i + 1
                    ].as_related_node_info()

        return nodes

    def _split_nodes(
            self, chunk_level: str, node: BaseNode
    ) -> List[BaseNode]:
        if chunk_level == 'document':
            nodes =  self._get_document_node_from_document(node)
        elif chunk_level == 'section':
            nodes =  self._get_section_nodes_from_document_node(node)
        elif chunk_level == 'paragraph':
            nodes = self._get_paragraph_nodes_from_section_node(node)
        elif chunk_level == 'multi-sentences':
            nodes = self._get_multiple_sentences_nodes_from_paragraph_node(node)
        else:
            raise Exception(f"Invalid chunk level {chunk_level}")
        
        nodes = self._postprocess_parsed_nodes(nodes, chunk_level)
        return nodes

    def _init_get_nodes_from_nodes(self, nodes):
        latest_level = 0
        prev_level_nodes = nodes

        if self._cache_process_path is not None:
            # init attributions
            latest_level = -1
            self._level2nodes = {}
            level2int = {level: i for i, level in enumerate(self._chunk_levels)}
            
            # loading nodes
            if os.path.exists(self._cache_process_path):
                file_size = os.path.getsize(self._cache_process_path)
                with open(self._cache_process_path, 'r') as cache_file:
                    with tqdm(total=file_size, desc='Loading cache...', unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                        for line in cache_file:
                            try:
                                node_dict = json.loads(line)
                                node = TextNode.from_dict(node_dict)
                                if level2int[node.metadata['level']] > latest_level:
                                    latest_level = level2int[node.metadata['level']]
                                    self._level2nodes[latest_level] = []
                                self._level2nodes[level2int[node.metadata['level']]].append(node)
                            except Exception as e:
                                print(e)
                            # Update progress bar based on bytes read
                            pbar.update(len(line))
            self._cache_process_file = open(self._cache_process_path, 'a+')
            latest_level = max(latest_level, 0)

            # get prev_level_nodes
            if latest_level == 1:
                latest_level = 0
                processed_prev_level_nodes_id = set()
                for node in self._level2nodes[latest_level]:
                    processed_prev_level_nodes_id.add(node.ref_doc_id)
                prev_level_nodes = []
                for node in nodes:
                    if node.id_ not in processed_prev_level_nodes_id:
                        prev_level_nodes.append(node)
                
            if latest_level > 1:
                processed_prev_level_nodes_id = set()
                for node in self._level2nodes[latest_level]:
                    parent_node_id = node.relationships[NodeRelationship.PARENT].node_id
                    processed_prev_level_nodes_id.add(parent_node_id)
                prev_level_nodes = []
                for node in self._level2nodes[latest_level-1]:
                    if node.id_ not in processed_prev_level_nodes_id:
                        prev_level_nodes.append(node)

        return latest_level, prev_level_nodes

    def _get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""
        latest_level, prev_level_nodes = self._init_get_nodes_from_nodes(nodes)

        for level in range(latest_level, len(self._chunk_levels)):
            if level not in self._level2nodes:
                self._level2nodes[level] = []

            # first split current nodes into sub-nodes
            nodes_with_progress = get_tqdm_iterable(
                prev_level_nodes, show_progress, f'{self._chunk_levels[level]} level parsing ...'
            )
            
            for node in nodes_with_progress:
                if node.metadata.get('level', None) == 'document' and 'isNew' not in node.metadata:
                    continue
                cur_sub_nodes = self._split_nodes(chunk_level=self._chunk_levels[level], node=node)

                # add parent relationship from sub node to parent node
                # add child relationship from parent node to sub node
                # NOTE: Only add relationships if level > 0, since we don't want to add
                # relationships for the top-level document objects that we are splitting
                if level == 0:
                    cur_sub_nodes[0].metadata['isNew'] = True
                elif level > 0:
                    for sub_node in cur_sub_nodes:
                        _add_parent_child_relationship(
                            parent_node=node,
                            child_node=sub_node,
                        )

                if level == 1:
                    del node.metadata['isNew']
                    self.save_nodes([node] + cur_sub_nodes)
                elif level > 1:
                    self.save_nodes(cur_sub_nodes)
                
                self._level2nodes[level].extend(cur_sub_nodes)
            prev_level_nodes = self._level2nodes[level]

        if self._cache_process_path is not None:
            self._cache_process_file.close()

        all_nodes = []
        for nodes in self._level2nodes.values():
            all_nodes.extend(nodes)
        return all_nodes

    def get_nodes_from_documents(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse documents into nodes.

        Args:
            documents (Sequence[Document]): documents to parse
            show_progress (bool): whether to show progress bar

        """
        self._doc_id_to_document = {doc.id_: doc for doc in documents}

        with self.callback_manager.event(
            CBEventType.NODE_PARSING, payload={EventPayload.DOCUMENTS: documents}
        ) as event:
            all_nodes = self._get_nodes_from_nodes(
                nodes=documents,
                show_progress=show_progress
            )

            event.on_end({EventPayload.NODES: all_nodes})

        return all_nodes
    
    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)

class ManuallyParser():
    def __init__(self, cache_path, cache_name, delete_cache=True, force=False) -> None:
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.delete_cache = delete_cache
        self.force = force

    def get_nodes_from_documents(self, nodes, **kwargs):
        nodes_cache_path = os.path.join(self.cache_path, f"{self.cache_name}_finished.json")
        if os.path.exists(nodes_cache_path) and not self.force:
            nodes = load_nodes_jsonl(nodes_cache_path)
            if self.delete_cache:
                os.remove(nodes_cache_path)
            return nodes
        nodes_cache_path = os.path.join(self.cache_path, f"{self.cache_name}_{len(nodes)}_processing.jsonl")
        save_nodes_jsonl(nodes_cache_path, nodes)
        print(f"\n[Manually Parser] Cache \'{self.cache_name}\' has been saved. Waiting for processing manually...")
        exit()