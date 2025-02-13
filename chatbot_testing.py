import os #for the environment file or variables
from openai import OpenAI #for the LLM
from langchain_openai import OpenAIEmbeddings #to use semanticchunker textsplitter
from langchain_community.document_loaders import PyPDFLoader #to load documents
from langchain_experimental.text_splitter import SemanticChunker #to split texts
import chromadb #save split text documents for retrieval  

#manually export the keys because dotenv is satan

openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

#Load and preprocess documents
loader = PyPDFLoader("/Users/kyli/Downloads/Sleep Paper.pdf")
#Load documents into a variable called documents, which is a list of document objects
#Attributes of each document object are page_content and metadata
documents = loader.load()


#Split the text using Recursive Character Text Splitter - other text splitting algos available
text_splitter = SemanticChunker(OpenAIEmbeddings())

#Create a list of text splits
split_docs = text_splitter.split_documents(documents)

#Initialize database that gets wiped whenever
client = chromadb.Client()

#client = chromadb.PersistentClient(path="/path/to/save/to")
#Above for the final version
#https://docs.trychroma.com/docs/run-chroma/persistent-client

# Create a collection in Chroma (the actual database)
collection = client.get_or_create_collection(name="my_collection")
#Add each text chunk to the db
collection.add(
    documents=[doc.page_content for doc in split_docs],  # Extract text from splits
    ids=[f"doc_{i}" for i in range(len(split_docs))]  # Generate unique IDs
)

results = collection.query(
    query_texts=["This is a query document about outcome"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)
