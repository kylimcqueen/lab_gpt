import os #for the environment file or variables
from openai import OpenAI #for the LLM
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings #to use semanticchunker textsplitter
from langchain_community.document_loaders import PyPDFLoader #to load documents
from langchain_experimental.text_splitter import SemanticChunker #to split texts
import chromadb #save split text documents for retrieval  


#manually export the keys because dotenv is satan

openai_api_key = os.getenv("OPENAI_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#Load and preprocess documents
loader = PyPDFLoader("/Users/kyli/Desktop/SQAT-Manual.pdf")

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

#Instantiate API endpoint for converting text into vector representations - embeddings
client = OpenAI()


def get_embeddings(texts, model="text-embedding-ada-002"):
    """Generates embeddings for a list of texts using OpenAI."""
    return [
        client.embeddings.create(input=text, model=model).data[0].embedding
        for text in texts
    ]

# Extract text chunks from split documents
texts = [doc.page_content for doc in split_docs]
embeddings = get_embeddings(texts)

# Add embeddings to Chroma collection
collection.add(
    embeddings=embeddings,  # Add vector embeddings instead of raw text
    ids=[f"doc_{i}" for i in range(len(embeddings))]
)


chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat