from typing import Dict, List
from llama_index.core.llms import LLM
import threading
import concurrent.futures
from utils.evaluate_execution_time import aevaluate_time, evaluate_time

class TreeSummarize():
    def __init__(
            self, 
            query_str: str, 
            summary_str: str,
            qa_prompt: str, 
            llm: LLM, 
            refine_times: int
        ):
        # Normal synchronous initialization
        self.query_str: str = query_str
        self.summary_str: str = summary_str
        self.qa_prompt: str = qa_prompt
        self.llm: LLM = llm
        self.response_txt: str = None
        self.prompt_records: Dict = {}
        self.refine_times = refine_times

    @classmethod
    def from_defaults(
        cls,
        query_str: str, 
        summary_str: str,
        qa_prompt: str,
        llm: LLM,
        refine_times: int=10
    ):
        self = cls(query_str, summary_str, qa_prompt, llm, refine_times)
        return self
    
    def evaluate_response(self, i, p):
        with self.semaphore:
            with self._lock:
                self._running_tasks += 1
                print(f"Task {i} started. Running tasks: {self._running_tasks}")
            try:
                response, elapsed_time = evaluate_time(lambda : self.llm.complete(p))
                with self._lock:
                    self._running_tasks -= 1
                print(f"Task {i} completed: {elapsed_time:.2f} seconds. Running tasks: {self._running_tasks}")
                return response
            except Exception as e:
                print(f"Error occurred: {e}")
                with self._lock:
                    self._running_tasks -= 1
                return ""

    def combine_results(self, texts, level):
        self.prompt_records[level] = []
        for idx in range(0, len(texts), self.num_children):
            text_batch = texts[idx : idx + self.num_children]
            context_str = "\n\n".join([t for t in text_batch])
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=context_str, query_str=self.query_str
            )
            self.prompt_records[level].append(fmt_qa_prompt)

        # tasks = [self.llm.acomplete(p) for p in cur_prompt_list]
        # combined_responses = await asyncio.gather(*tasks)
        responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            print(f"prompt number: {len(self.prompt_records[level])}")
            for i, p in enumerate(self.prompt_records[level]):
                futures.append(executor.submit(self.evaluate_response, i=i, p=p))
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                responses.append(result)
        
        new_texts = [r.text.strip() for r in responses]

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
            new_text = str(self.llm.complete(fmt_qa_prompt)).strip()
            text = text if len(text) < len(new_text) else new_text
            i += 1
        return text

    def generate_response_hs(self, texts: List[str], num_children=10):
        """Generate a response using hierarchical summarization strategy.

        Combine num_children nodes hierarchically until we get one root node.

        """
        self.num_children = num_children
        self.prompt_records[0] = []

        for text in texts:
            fmt_qa_prompt = self.qa_prompt.format(
                context_str=text, query_str=self.query_str
            )
            self.prompt_records[0].append(fmt_qa_prompt)

        self._lock = threading.Lock()
        self._running_tasks = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            print(f"prompt number: {len(self.prompt_records[0])}")
            for i, p in enumerate(self.prompt_records[0]):
                futures.append(executor.submit(self.evaluate_response, i=i, p=p))
            
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        response_txt = self.combine_results([r.text.strip() for r in responses], 1)
        response_txt = self.refine_response(response_txt)

        return response_txt, self.prompt_records


