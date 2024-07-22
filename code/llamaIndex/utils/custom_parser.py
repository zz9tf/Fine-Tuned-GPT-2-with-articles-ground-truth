from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.schema import BaseNode, Document, NodeRelationship, MetadataMode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.llms import LLM
from llama_index.core.node_parser.relational.hierarchical import _add_parent_child_relationship
from utils.schema import TemplateSchema
from utils.response_synthesis import TreeSummarize

class CustomHierarchicalNodeParser(NodeParser):
    """Hierarchical node parser.

    Splits a document into a recursive hierarchy Nodes using a NodeParser.

    NOTE: this will return a hierarchy of nodes in a flat list, where there will be
    overlap between parent nodes (e.g. with a bigger chunk size), and child nodes
    per parent (e.g. with a smaller chunk size).
    """
    llm: LLM = Field(
        default=None,
        description=(
            "LLM model to be used for generating node summary content of \'document\' and \'section\' levels"
        )
    )
  
    # The chunk level to use when splitting documents: document, section, paragraph, multi-sentences
    _chunk_levels: List[str] = PrivateAttr()

    _doc_id_to_document: Dict[str, Document] = PrivateAttr()

    _sentences_splitter: SentenceSplitter = PrivateAttr()
    
    @classmethod
    def from_defaults(
        cls,
        llm: LLM,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "CustomHierarchicalNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        cls._chunk_levels = ["document", "section", "paragraph", "multi-sentences"]


        cls._sentences_splitter = SentenceSplitter(
            chunk_size=128, 
            chunk_overlap=20, 
            include_metadata=False, 
            include_prev_next_rel=False
        )

        return cls(
            llm=llm,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "CustomHierarchicalNodeParser"

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

    def _get_section_nodes_from_document_node(
        self, 
        document_node: BaseNode,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        parent_document = self._doc_id_to_document.get(document_node.ref_doc_id, None)
        if parent_document is None:
            raise Exception(f"Parent document is not found with id {document_node.ref_doc_id}")

        titles = []
        sections = []
        summaries = []
        abstract = None
        for title, (start, end) in parent_document.metadata['sections'].items():
            section = document_node.get_content()[start: end]

            if title == 'abstract':
                # Tree summary
                texts = section.split('\n')
                abstract, _ = TreeSummarize.from_defaults(
                    texts=texts,
                    query_str=TemplateSchema.tree_summary_section_q_Tmpl,
                    summary_str=TemplateSchema.tree_summary_summary_Tmpl,
                    qa_prompt=TemplateSchema.tree_summary_qa_Tmpl,
                    llm=self.llm,
                    num_children=3
                ).generate_response_hs()

            else:
                titles.append(title)
                sections.append(section)
                # Tree summary
                texts = section.split('\n')
                summary, _ = TreeSummarize.from_defaults(
                    texts=texts,
                    query_str=TemplateSchema.tree_summary_section_q_Tmpl,
                    summary_str=TemplateSchema.tree_summary_summary_Tmpl,
                    qa_prompt=TemplateSchema.tree_summary_qa_Tmpl,
                    llm=self.llm,
                    num_children=3
                ).generate_response_hs()
                summaries.append(summary)

        summary_for_document = None

        if abstract != None:
            summary_for_document = abstract
        else:
            # Tree summary
            summary_for_document, _ = TreeSummarize.from_defaults(
                texts=summaries,
                query_str=TemplateSchema.tree_summary_section_q_Tmpl,
                summary_str=TemplateSchema.tree_summary_summary_Tmpl,
                qa_prompt=TemplateSchema.tree_summary_qa_Tmpl,
                llm=self.llm,
                num_children=3
            ).generate_response_hs()

        # Update document node
        origin_document_text = document_node.text
        document_node.metadata['original_content'] = origin_document_text
        document_node.text = summary_for_document

        # Update section nodes
        all_nodes.extend(
            build_nodes_from_splits(summaries, document_node, id_func=self.id_func)
        )

        for title, node in zip(titles, all_nodes):
            node.metadata['section_title'] = title
            node.metadata['original_content'] = section

        return all_nodes

    def _get_paragraph_nodes_from_section_node(
        self,
        section_node: BaseNode
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        
        splits = section_node.get_content().split('\n')

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
        exit()
        return self._sentences_splitter._parse_nodes([paragraph_node])

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

    def _recursively_get_nodes_from_nodes(
        self,
        nodes: List[BaseNode],
        level: int,
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Recursively get nodes from nodes."""
        if level >= len(self._chunk_levels):
            raise ValueError(
                f"Level {level} is greater than number of text "
                f"splitters ({len(self._chunk_levels)})."
            )

        # first split current nodes into sub-nodes
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, f'{self._chunk_levels[level]} level parsing ...'
        )
        sub_nodes = []
        for node in nodes_with_progress:
            cur_sub_nodes = self._split_nodes(chunk_level=self._chunk_levels[level], node=node)

            # add parent relationship from sub node to parent node
            # add child relationship from parent node to sub node
            # NOTE: Only add relationships if level > 0, since we don't want to add
            # relationships for the top-level document objects that we are splitting
            if level > 0:
                for sub_node in cur_sub_nodes:
                    _add_parent_child_relationship(
                        parent_node=node,
                        child_node=sub_node,
                    )

            sub_nodes.extend(cur_sub_nodes)

        # now for each sub-node, recursively split into sub-sub-nodes, and add
        if level < len(self._chunk_levels) - 1:
            sub_sub_nodes = self._recursively_get_nodes_from_nodes(
                sub_nodes,
                level + 1,
                show_progress=show_progress,
            )
        else:
            sub_sub_nodes = []

        return sub_nodes + sub_sub_nodes

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
            
            all_nodes = self._recursively_get_nodes_from_nodes(
                nodes=documents,
                level=0,
                show_progress=show_progress
            )

            event.on_end({EventPayload.NODES: all_nodes})

        return all_nodes
    
    # Unused abstract method
    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        return list(nodes)