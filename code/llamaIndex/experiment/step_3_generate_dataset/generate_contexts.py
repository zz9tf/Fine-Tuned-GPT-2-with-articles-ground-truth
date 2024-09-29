

if __name__ == '__main__':
    retriever = index.as_retriever(
        retriever_mode="llm",
    )