import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from update_database import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Question:
{question}
---
Requirements:
1. Repeat and answer the question first. 
2. Judge if additional context is useful or not(No suggestions it is useful or not). Explain why.
---
Additional context
{context}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # references
    sources = []
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for i, (doc, score) in enumerate(sorted_results):
        sources.append("[{}] score: {} path: {}".format(i, score, doc.metadata.get("id", None)))

    sources = "\n".join(sources)
    formatted_response = f"Response:\n>>> {response_text}\nSources:\n{sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()