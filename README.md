# Retrieval-Augmented Generation (RAG) with Semantic Search using ChromaDB for Insurance Policies

## Overview
This repository demonstrates a pipeline for Retrieval-Augmented Generation (RAG) applied to the insurance domain. The system enables efficient retrieval and generation of information from insurance-related documents, such as policy documents, claims records, and regulatory guidelines. It leverages OpenAI's text embedding model, ChromaDB for vector storage, and a cross-encoder for reranking. The approach integrates PDF processing, semantic search with caching, and re-ranking, leading to more accurate and contextually relevant responses.

## Table of Contents
- [Installation](#installation)
- [PDF Processing](#pdf-processing)
- [Generating and Storing Embeddings](#generating-and-storing-embeddings)
- [Semantic Search with Cache](#semantic-search-with-cache)
- [Re-Ranking with a Cross Encoder](#re-ranking-with-a-cross-encoder)
- [Retrieval-Augmented Generation](#retrieval-augmented-generation)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Conclusions](#conclusions)
- [Acknowledgements](#acknowledgements)

## Installation
Install all the required libraries:
```bash
pip install -U pdfplumber tiktoken openai chromadb sentence-transformers
```

Import the necessary libraries:
```python
import pdfplumber
from pathlib import Path
import pandas as pd
import json
import tiktoken
import openai
import chromadb
```

## PDF Processing
### Extracting text from a PDF
This project uses `pdfplumber` to extract text and tables from PDFs efficiently.
```python
with pdfplumber.open('sample.pdf') as pdf:
    page = pdf.pages[0]
    text = page.extract_text()
    print(text)
```
### Extracting text from multiple PDFs
A function to extract text from multiple PDFs and store them in a DataFrame.
```python
def extract_text_from_pdf(pdf_path):
    extracted_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            extracted_text.append([f'Page {i+1}', page.extract_text()])
    return extracted_text
```

## Generating and Storing Embeddings
Using OpenAI's `text-embedding-ada-002` model to generate embeddings and store them in ChromaDB.
```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-ada-002")
client = chromadb.PersistentClient(path='rag_chromadb')

collection = client.get_or_create_collection(name='Insurance_Policies', embedding_function=embedding_function)
```

## Semantic Search with Cache
Using ChromaDB for efficient semantic search with caching.
```python
def semantic_search(query, n_results=10):
    results = collection.query(query_texts=[query], n_results=n_results)
    return results
```

## Re-Ranking with a Cross Encoder
Re-ranking results using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query, results):
    pairs = [[query, result] for result in results['documents'][0]]
    scores = cross_encoder.predict(pairs)
    ranked_results = sorted(zip(results['documents'][0], scores), key=lambda x: x[1], reverse=True)
    return ranked_results
```

## Retrieval-Augmented Generation
Combining the retrieved documents with GPT-3.5 to generate a response.
```python
def generate_response(query, top_results):
    messages = [
        {"role": "system", "content": "You are an expert in insurance policies and claims."},
        {"role": "user", "content": f"Based on the following insurance documents, answer: {query}\n{top_results}"}
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return response["choices"][0]["message"]["content"]
```

## Results
The pipeline enhances response accuracy for insurance-related queries by integrating document retrieval with advanced LLM processing. It is particularly useful for policy document retrieval, claims analysis, and regulatory compliance assistance.

## Technologies Used
- Python
- OpenAI API
- pdfplumber
- ChromaDB
- Sentence Transformers

## Conclusions
This approach improves the efficiency and accuracy of retrieving and generating insights from insurance-related documents. It facilitates better decision-making in insurance underwriting, claims management, and policy analysis.

## Acknowledgements
Thanks to OpenAI, ChromaDB, Upgrad and the NLP community for their contributions.

