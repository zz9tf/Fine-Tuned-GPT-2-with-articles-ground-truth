import pandas as pd
import numpy as np
import re
from PyPDF2 import PdfReader
import os
import docx
from tqdm import tqdm

def read_pdf(file_path):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

def read_word(file_path):
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file_path):
    with open(file_path, "r") as file:
        text = file.read()
    return text

def read_documents_from_directory(directory):
    combined_text = []
    for filename in tqdm(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        try:
            if filename.endswith(".pdf"):
                new_text = read_pdf(file_path)
            elif filename.endswith(".docx"):
                new_text = read_word(file_path)
            elif filename.endswith(".txt"):
                new_text = read_txt(file_path)
            # new_text = re.sub(r'\n+', '\n', new_text).strip()
            combined_text.append(new_text)
        except Exception as e:
            print(str(e))
    return combined_text