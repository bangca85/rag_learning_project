import os
import sys
from dotenv import load_dotenv

# Make sure we are in the correct directory to load the .env file
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Load environment variables from .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)  # Explicitly loading the .env

# Check if the key is loaded
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found in environment variables!")

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = openai_api_key

from helper_functions import *
from evaluation.evalute_rag import *

path = "../data/Accelerate - Building and Scaling High Performing Technology Organisations - Nicole Fergrson.pdf"

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    # Encodes PDF into vector store
    loader = PyPDFLoader(path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
    
    return vectorstore

chunks_vector_store = encode_pdf(path, chunk_size=1000, chunk_overlap=200)
chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 2})

test_query = "What is the main cause of climate change?"
context = retrieve_context_per_question(test_query, chunks_query_retriever)
show_context(context)

evaluate_rag(chunks_query_retriever)
