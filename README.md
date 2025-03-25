# Retrieval-Augmented Generation (RAG) Pipeline(PDF Processing) using ChromaDB for Insurance Domain

## Overview
This project provides an end-to-end solution for processing PDF files, extracting text and tables, storing and embedding them in a vector database, and performing retrieval-augmented generation (RAG) using OpenAI's embeddings and ChromaDB.

## Table of Contents
- [Installation](#installation)
- [PDF Processing](#pdf-processing)
- [Generating and Storing Embeddings](#generating-and-storing-embeddings)
- [Semantic Search with Cache](#semantic-search-with-cache)
- [Re-Ranking with a Cross Encoder](#re-ranking-with-a-cross-encoder)
- [Retrieval-Augmented Generation](#retrieval-augmented-generation)
- [Results](#results)

## Installation
Ensure you have the required libraries installed before running the code:
```bash
pip install -U pdfplumber tiktoken openai chromaDB sentence-transformers
```

## PDF Processing
### Reading and Extracting Text from PDFs
We use `pdfplumber` to read PDF files, extract text, and identify tables. This helps us preprocess documents for further analysis.

#### Reading a Single PDF File
```python
import pdfplumber

pdf_path = "path/to/pdf.pdf"
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[6]  # Extract a specific page
    text = page.extract_text()
    tables = page.extract_tables()
    print(text, tables)
```

#### Extracting Text from Multiple PDFs
We define a function to extract and structure text from multiple PDFs.
```python
from pathlib import Path
import pandas as pd
import json

def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            extracted_text.append([f"Page {i+1}", text])
    return extracted_text

pdf_directory = Path("/path/to/pdf/folder")
data = []
for pdf_file in pdf_directory.glob("*.pdf"):
    extracted_text = extract_text_from_pdf(pdf_file)
    df = pd.DataFrame(extracted_text, columns=['Page No.', 'Text'])
    df['Document Name'] = pdf_file.name
    data.append(df)

all_pdfs_data = pd.concat(data, ignore_index=True)
```

## Generating and Storing Embeddings
### Using OpenAI for Text Embeddings
We use OpenAI’s `text-embedding-ada-002` model to convert text into embeddings and store them in a ChromaDB collection.
```python
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

openai.api_key = "your_openai_api_key"
client = chromadb.PersistentClient(path="/path/to/chromadb")
embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002")

collection = client.get_or_create_collection(name='InsuranceDocs', embedding_function=embedding_function)
collection.add(documents=all_pdfs_data['Text'].tolist(),
               ids=[str(i) for i in range(len(all_pdfs_data))],
               metadatas=all_pdfs_data[['Page No.', 'Document Name']].to_dict(orient='records'))
```

## Semantic Search with Cache
To optimize retrieval, we first search a cache and then the main collection.
```python
query = input("Enter your search query: ")
cache_collection = client.get_or_create_collection(name='InsuranceCache', embedding_function=embedding_function)

cache_results = cache_collection.query(query_texts=[query], n_results=1)
if not cache_results['documents'][0]:
    results = collection.query(query_texts=[query], n_results=10)
    cache_collection.add(documents=[query], ids=[query], metadatas={'results': results})
    print("Results from main collection:", results)
else:
    print("Results from cache:", cache_results['documents'])
```

## Re-Ranking with a Cross Encoder
To improve ranking, we use a cross-encoder model from `sentence-transformers`.
```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

scores = cross_encoder.predict([[query, doc] for doc in results['documents'][0]])
results_df['Re-rank Score'] = scores
results_df = results_df.sort_values(by='Re-rank Score', ascending=False)
```

## Retrieval-Augmented Generation
Finally, we pass top-ranked results to GPT-3.5 to generate an answer.
```python
def generate_response(query, results_df):
    messages = [
        {"role": "system", "content": "You are an insurance assistant."},
        {"role": "user", "content": f"Answer the query: {query} using {results_df.head(3).to_dict()}"}
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return response["choices"][0]["message"]["content"]

print(generate_response(query, results_df))
```

## Results
The system retrieves relevant pages, ranks them, and generates responses using OpenAI’s GPT-3.5. This enhances document search capabilities, especially for structured text like insurance policies.

---
This project enables efficient retrieval and summarization of PDF documents using advanced NLP techniques.
