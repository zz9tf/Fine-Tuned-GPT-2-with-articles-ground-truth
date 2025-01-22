import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import gc
import torch
from typing import Dict, List
from component.models.llm.get_llm import get_llm
from utils.evaluate_execution_time import evaluate_time

class TreeSummarize():
    def __init__(
            self, 
            query_str: str, 
            summary_str: str,
            qa_prompt: str, 
            llm_config: dict, 
            refine_times: int
        ):
        # Normal synchronous initialization
        self.query_str: str = query_str
        self.summary_str: str = summary_str
        self.qa_prompt: str = qa_prompt
        self.llm_config = llm_config
        # self.llm: LLM = get_llm(self.llm_self, self.llm_config)
        self.llm = None
        self.response_txt: str = None
        self.prompt_records: Dict = {}
        self.refine_times = refine_times

    @classmethod
    def from_defaults(
        cls,
        query_str: str, 
        summary_str: str,
        qa_prompt: str,
        llm_config: dict,
        refine_times: int=10
    ):
        self = cls(query_str, summary_str, qa_prompt, llm_config, refine_times)
        return self
    
    def del_llm(self):
        if hasattr(self, 'llm'):
            del self.llm
            torch.cuda.empty_cache()
            gc.collect()
    
    def load_llm(self):
        self.llm = get_llm(self.llm_config)

    def evaluate_response(self, i, p):
        response, elapsed_time = evaluate_time(lambda : self.llm.complete(p))
        print(f"Task {i} completed: {elapsed_time:.2f} seconds. Running tasks: {self._running_tasks}")
        return response

    def combine_results(self, texts, level) -> str:
        assert len(texts) > 0, "[Wrong] Invalid texts with length 0"

        self.prompt_records[level] = []
        responses = []
        for idx in range(0, len(texts), self.num_children):
            text_batch = texts[idx : idx + self.num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=self.query_str
            )
            self.prompt_records[level].append(fmt_qa_prompt)
            with torch.no_grad():
                try:
                    response = self.llm.complete(fmt_qa_prompt)
                except Exception as e:
                    print(f"text: {len(''.join(text_batch))}\ntext batch number: {len(text_batch)}\nquery_str: {len(self.query_str)}\nprompt: {len(fmt_qa_prompt)}")
                    print()
                    print(e)
                    exit()

            torch.cuda.empty_cache()
            gc.collect()
            responses.append(response)
        
        new_texts = [r.strip() for r in responses]

        assert len(new_texts) != 0, "[Wrong] Invalid combine results that the length of new_texts is 0"

        if len(new_texts) == 1:
            return new_texts[0]
        else:
            return self.combine_results(new_texts, level+1)

    def refine_response(self, text):
        i = 0
        while len(text) > 500 and i < self.refine_times:
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=text, query_str=self.summary_str
            )
            # print(fmt_qa_prompt)
            with torch.no_grad():
                response = self.llm.complete(fmt_qa_prompt)
            torch.cuda.empty_cache()
            gc.collect()
            new_text = str(response).strip()
            text = text if len(text) < len(new_text) else new_text
            i += 1
        return text

    def generate_response_hs(self, texts: List[str], num_children=10):
        """Generate a response using hierarchical summarization strategy.

        Combine num_children nodes hierarchically until we get one root node.

        """
        # print(f"texts: \n{texts}")
        assert len(texts) > 0, "[Wrong] Invalid texts with length 0"
        self.num_children = num_children
        self.prompt_records[0] = []
        responses = []

        for text in texts:
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=text, query_str=self.query_str
            )
            self.prompt_records[0].append(fmt_qa_prompt)
            if len(fmt_qa_prompt) > 8000:
                print("at stage 8000")
                print(f"text: {len(text)}\nquery_str: {len(self.query_str)}\nprompt: {len(fmt_qa_prompt)}")
                print(fmt_qa_prompt)
                print()
            with torch.no_grad():
                try:
                    response = self.llm.complete(fmt_qa_prompt)
                except Exception as e:
                    print(f"text: {len(text)}\nquery_str: {len(self.query_str)}\nprompt: {len(fmt_qa_prompt)}")
                    print(fmt_qa_prompt)
                    print()
                    print(f'error message: {e}')
                    exit()
                
            torch.cuda.empty_cache()
            gc.collect()
            responses.append(response)

        response_txt = self.combine_results([r.strip() for r in responses], 1)
        # response_txt = self.refine_response(response_txt)

        return response_txt, self.prompt_records