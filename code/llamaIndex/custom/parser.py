import os, sys
sys.path.insert(0, os.path.abspath('..'))
import io
import uuid
import json
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence
from tqdm import tqdm
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.schema import BaseNode, Document, NodeRelationship, MetadataMode, TextNode
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.llms import LLM
from custom.io import load_nodes_jsonl, save_nodes_jsonl
from custom.schema import TemplateSchema
from custom.response_synthesis import TreeSummarize
from custom.semantic import SemanticSplitter

##########################################################################
# parser
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SimpleFileNodeParser, 
    HierarchicalNodeParser
)

def get_parser(self, config, **kwargs):
    """get a parser"""
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
        raise Exception(
            f"Invalid parser config with config {config}. Please provide parser types {self.prefix_config['parser'].keys()}"
        )
    
##########################################################################

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

    _sentences_splitter: SentenceSplitter = PrivateAttr()
    
    _embedding_config: Dict = PrivateAttr()

    _llm_self = PrivateAttr()

    _llm_config: Dict = PrivateAttr()
    
    _tree_summarizer: TreeSummarize = PrivateAttr()

    _semantic_splitter: SemanticSplitter = PrivateAttr()

    @classmethod
    def from_defaults(
        cls,
        llm_self,
        llm_config: LLM,
        embedding_config: BaseEmbedding,
        cache_dir_path: str = os.path.abspath('.'),
        cache_file_name: str = 'CustomHierarchicalNodeParser_cache.jsonl',
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None
    ) -> "CustomHierarchicalNodeParser":
        """get a CustomHierarchicalNodeParser"""
        callback_manager = callback_manager or CallbackManager([])
        cls._chunk_levels = ["document", "section", "paragraph", "multi-sentences"]
        
        cls._cache_process_file = None
        cls._cache_process_path = os.path.join(cache_dir_path, cache_file_name)

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
        cls._semantic_splitter = SemanticSplitter(buffer_size=1, breakpoint_percentile_threshold=97, embed_model_config=embedding_config)

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
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
            chunk_size=1536, 
            chunk_overlap=20, 
            include_metadata=False, 
            include_prev_next_rel=False
        )
        force_splitter = SentenceSplitter(
            chunk_size=768, 
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
                    # paragraphs.extend(ps)
                    final_ps = []
                    for p_new in ps:
                        final_new_ps = force_splitter.split_text_metadata_aware(p_new, '')
                        final_ps.extend([p for p in final_new_ps if len(p.strip()) > 0])
                    paragraphs.extend(final_ps)
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

    def _add_start_char_idx_and_end_char_idx(self, child: BaseNode, parent: BaseNode):
        start_char_idx = parent.text.find(
            child.get_content(metadata_mode=MetadataMode.NONE)
        )

        # update start/end char idx
        if start_char_idx >= 0:
            child.start_char_idx = start_char_idx
            child.end_char_idx = start_char_idx + len(
                child.get_content(metadata_mode=MetadataMode.NONE)
            )

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
        self._add_start_char_idx_and_end_char_idx(document_node, document)
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

    def _add_parent_child_relationship(self, child: BaseNode, parent: BaseNode) -> None:
        """Add parent/child relationship between nodes."""
        child_list = parent.relationships.get(NodeRelationship.CHILD, [])
        child_list.append(child.as_related_node_info())

        parent.relationships[NodeRelationship.CHILD] = child_list
        child.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()

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
            self._add_start_char_idx_and_end_char_idx(node, document_node)
            # Add relationships
            self._add_previous_and_next_relationship(i, all_nodes, node)
            self._add_parent_child_relationship(node, document_node)
            self._update_exclude_keys(node)

        return all_nodes

    def filter_lines(self, content):
        """remove all lines smaller than 50"""
        for line in content.split('\n'):
            if len(line.strip()) > 50:
                yield line

    def _get_paragraph_nodes_from_section_node(
        self,
        section_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        original_content = section_node.metadata['original_content']
        splits = list(self.filter_lines(original_content))
        all_nodes.extend(
            build_nodes_from_splits(splits, section_node, id_func=self.id_func)
        )
        for i, node in enumerate(all_nodes):
            # Update metadata
            node.metadata['level'] = 'paragraph'    
            self._add_start_char_idx_and_end_char_idx(node, section_node)
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
            self._add_start_char_idx_and_end_char_idx(node, paragraph_node)
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
    
    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)

class ManuallyParser():
    """generate cache for mannually parser"""
    def __init__(self, cache_path, cache_name, delete_cache=True, force=False) -> None:
        self.cache_name = cache_name
        self.cache_path = cache_path
        self.delete_cache = delete_cache
        self.force = force

    def get_nodes_from_documents(self, nodes, **kwargs):
        """get nodes from documents"""
        nodes_cache_path = os.path.join(self.cache_path, f"{self.cache_name}_finished.json")
        if os.path.exists(nodes_cache_path) and not self.force:
            nodes = load_nodes_jsonl(nodes_cache_path)
            if self.delete_cache:
                os.remove(nodes_cache_path)
            return nodes
        nodes_cache_path = os.path.join(
            self.cache_path, f"{self.cache_name}_{len(nodes)}_processing.jsonl"
        )
        save_nodes_jsonl(nodes_cache_path, nodes)
        print(
            f"\n[Manually Parser] Cache \'{self.cache_name}\' has been saved." +\
            "Waiting for processing manually..."
        )
        sys.exit(0)
