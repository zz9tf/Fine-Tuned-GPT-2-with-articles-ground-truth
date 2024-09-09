import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma

from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings

def load_documents(data_path):
    document_loader = PyPDFDirectoryLoader(data_path)

    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0
    print("calculating chunk id...")
    from tqdm import tqdm
    for chunk in tqdm(chunks):
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def add_to_database(path, CHROMA_PATH):
    def add_to_chroma(chroma_path, chunks: list[Document]):
        # Load the existing database.
        # Chroma: vactor database
        db = Chroma(
            persist_directory=chroma_path, embedding_function=get_embedding_function()
        )

        # Calculate Page IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        print("adding new chunks...")
        from tqdm import tqdm
        for chunk in tqdm(chunks_with_ids):
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")
    DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), path))

    # Create (or update) the data store.
    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    add_to_chroma(CHROMA_PATH, chunks)

def clear_database(chroma_path):
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

def main():
    path = "../../data/"
    CHROMA_PATH = "chroma"
    recorded_log_path = os.path.join(path, "recorded_log.txt")
    recorded_log = None
    recorded_files = set()
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database(CHROMA_PATH)
        recorded_log =  open(recorded_log_path, "w+")
        recorded_log.seek(0)
    else:
        recorded_log =  open(recorded_log_path, "a+")
        recorded_log.seek(0)
        recorded_files = set(recorded_log.read().split("\n"))
    n = len(os.listdir(path))
    print(recorded_files)
    for i, file in enumerate(os.listdir(path)):
        if file.endswith(".pdf") and file not in recorded_files:
            folder_name = file.split(".")[0]
            folder_path = os.path.join(os.path.join(path, folder_name))
            os.makedirs(folder_path)
            file_path = os.path.join(path, file)
            shutil.copy(file_path, folder_path)
            print(folder_path, "   {}/{} | {:.2f}%".format(i+1, n, 100*(i+1)*1.0/n))
            try:
                add_to_database(folder_path, CHROMA_PATH)
            except Exception as e:
                print("-------------------------------")
                print(e)
                print("-------------------------------")
            recorded_log.write(file+"\n")
            shutil.rmtree(folder_path)

    
if __name__ == "__main__":
    main()