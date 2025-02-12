import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader


#manually export the keys because dotenv is satan

openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

#Load and preprocess documents
loader = PyPDFLoader("/Users/kyli/Desktop/lab_gpt_example_docs/01_pathophys_tbi.pdf")