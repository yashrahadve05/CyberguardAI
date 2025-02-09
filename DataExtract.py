import json
import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = '\n'.join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def text_to_json(text, json_path='output.json'):
    """Convert extracted text to a JSON file."""
    data = {"content": text}
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=2)

def json_to_chunks(json_path, chunk_size=200, overlap=50):
    """Split JSON content into chunks using langchain's RecursiveCharacterTextSplitter."""
    with open(json_path, 'r') as file:
        data = json.load(file)
        text = data['content']
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def chunks_to_openai_embeddings(chunks, api_key):
    """Convert chunks to embeddings using OpenAI's text-embedding-ada-002 model."""
    os.environ['OPENAI_API_KEY'] = api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Store embeddings in FAISS for efficient retrieval
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def main():
    pdf_path = "DataSet/Cyber_crime_and_its_classification.pdf"
    json_path = "output.json"
    chunk_size = 200
    overlap = 50
    
    # Define your OpenAI API key here
    OPENAI_API_KEY = "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"
    
    # Step 1: Extract text from PDF
    print("Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Step 2: Convert text to JSON
    print("Converting text to JSON...")
    text_to_json(extracted_text, json_path)
    
    # Step 3: Split JSON into chunks
    print("Splitting JSON into chunks...")
    chunks = json_to_chunks(json_path, chunk_size, overlap)
    
    # Step 4: Convert chunks to embeddings using OpenAI's embedding model
    print("Generating embeddings using OpenAI...")
    vectorstore = chunks_to_openai_embeddings(chunks, OPENAI_API_KEY)
    
    print("Embeddings stored successfully in FAISS.")
    
if __name__ == "__main__":
    main()
    