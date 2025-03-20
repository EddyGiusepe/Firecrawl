#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro
"""
# 1. Load website with Firecrawl
from langchain_community.document_loaders import FireCrawlLoader  # Importing the FirecrawlLoader

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

url = "https://firecrawl.dev"

loader = FireCrawlLoader(
    api_key=firecrawl_api_key, # Note: Replace 'YOUR_API_KEY' with your actual FireCrawl API key
    url=url,  # Target URL to crawl
    mode="crawl"  # Mode set to 'crawl' to crawl all accessible subpages
)

docs = loader.load()
print(len(docs))
print(docs[0])

# 2. Setup the Vectorstore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
splits = splits[:50]
len(splits)
vectorstore = FAISS.from_documents(documents=splits,
                                   embedding=OllamaEmbeddings(model="jeffh/intfloat-multilingual-e5-large:f16"))

# 3. Retrieval and Generation
question = "O que é firecrawl?"
docs = vectorstore.similarity_search(query=question)
print(docs)

# 4. Generation
from groq import Groq

client = Groq(
    api_key=groq_api_key,
)

completion = client.chat.completions.create(
    model="llama3-8b-8192", # https://console.groq.com/docs/models
    messages=[
        {
            "role": "user",
            "content": f"""Você é um assistente amigável. Seu objetivo é responder à pergunta do usuário
                           com base na documentação fornecida abaixo:\nDocs:\n\n{docs}\n\nQuestion: {question}
                           Responda sempre em português brasileiro (pt-br).
                        """
        }
    ],
    temperature=0.0,
    max_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

print(completion.choices[0].message.content)
