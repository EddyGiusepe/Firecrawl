#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

JSON Strict Mode OpenAI
=======================
Obter dados estruturados de LLMs é muito útil para desenvolvedores que 
integram IA em seus aplicativos, permitindo análise e processamento 
mais confiáveis ​​de saídas de modelos.
A OpenAI acaba de lançar novas versões do gpt-4o e gpt-4o-mini que 
incluem grandes melhorias para desenvolvedores que buscam obter dados 
estruturados de LLMs. Com a introdução de Saídas Estruturadas e Modo 
JSON Estrito, os desenvolvedores agora podem garantir uma saída JSON 
100% do tempo ao definir strict como true.

Running this code
-----------------
uv run OpenAI_Structured_Outputs_and_JSON_Strict_Mode.py
"""
# Etapa 1: inicializar o FirecrawlApp e o cliente OpenAI
import os
from firecrawl import FirecrawlApp
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
Eddy_key_openai  = os.environ['OPENAI_API_KEY']
Eddy_key_firecrawl = os.environ['FIRECRAWL_API_KEY']


#firecrawl_app = FirecrawlApp(api_key=Eddy_key_firecrawl)
client = OpenAI(api_key=Eddy_key_openai)

# Etapa 2: extrair dados de uma página da Web
url = 'https://mendable.ai'
#scraped_data = firecrawl_app.scrape_url(url)
response = requests.get(url)
soup =BeautifulSoup(response.text, "html.parser")
scraped_data = {
    'title': soup.title.string if soup.title else '',
    'paragraphs': [p.text for p in soup.find_all('p')],
    'links': [a['href'] for a in soup.find_all('a', href=True)],
    'headings': [h.text for h in soup.find_all(['h1', 'h2', 'h3'])]
}

# Etapa 3: Defina a solicitação da API OpenAI
messages = [
    {
        "role": "system",
        "content": """Você é um assistente útil que extrai dados estruturados de páginas da web.
                      Sempre responda em português brasileiro (pt-br).
                   """
    },
    {
        "role": "user",
        #"content": f"Extraia o título e a descrição do seguinte conteúdo HTML: {scraped_data['content']}"
        "content": f"""Extraia o título e a descrição do seguinte conteúdo HTML: {scraped_data}
                       Sempre responda em português brasileiro (pt-br).
                    """
    }
]

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "extracted_data",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string"
                },
                "description": {
                    "type": "string"
                }
            },
            "required": ["headline", "description"],
            "additionalProperties": False
        }
    }
}

# Etapa 4: chame a API OpenAI e extraia dados estruturados
# Se você está se perguntando quais modelos pode usar com a saída estruturada do OpenAI e o modo JSON Strict, eles são gpt-4o-2024-08-06 e gpt-4o-mini-2024-07-18.
chat_completion = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=messages,
    response_format=response_format
)

extracted_data = chat_completion.choices[0].message.content

print(extracted_data)
