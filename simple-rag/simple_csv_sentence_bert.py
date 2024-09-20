from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Load environment variables from .env file
load_dotenv()

# Step 1: Load the CSV file
file_path = 'data/customers-100.csv'
df = pd.read_csv(file_path)

# Step 2: Combine relevant fields into a single text column for embedding
df['combined_text'] = df['First Name'] + ' ' + df['Last Name'] + ', works for ' + df['Company'] + '. Located in ' + df['City'] + ', ' + df['Country'] + '. Contact: ' + df['Email'] + '.'

# Step 3: Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained model from Hugging Face

# Step 4: Generate embeddings for the combined text
doc_texts = df['combined_text'].tolist()  # Convert the combined text column to a list
text_vectors = model.encode(doc_texts, convert_to_tensor=True)

# Step 5: Move the tensor to CPU and convert to a numpy array
vector_matrix = text_vectors.cpu().numpy()

# Step 6: Create the FAISS index (Using L2 distance-based search)
index = faiss.IndexFlatL2(vector_matrix.shape[1])

# Step 7: Add the vectors to the FAISS index
index.add(vector_matrix)

# Create a simple class to wrap the docstore
class SimpleDocstore:
    def __init__(self, docs):
        self.docs = docs

    def search(self, doc_id):
        # Debugging: Check if we are accessing the correct document ID
        print(f"Docstore search for doc_id: {doc_id}")
        return self.docs.get(int(doc_id), "ID not found")

# Step 8: Initialize the SimpleDocstore with the document texts wrapped in Document objects
docstore = SimpleDocstore({i: Document(page_content=doc_texts[i]) for i in range(len(doc_texts))})

# Create a mapping from index to document IDs
index_to_docstore_id = {i: i for i in range(len(doc_texts))}

# Debugging: Check the mapping
print("Index to Docstore ID Mapping:")
for i, doc_id in index_to_docstore_id.items():
    document = docstore.search(doc_id)
    if document:
        print(f"Index: {i}, Docstore ID: {doc_id}, Document: {document.page_content}")
    else:
        print(f"Document not found for Docstore ID: {doc_id}")

# Step 9: Define the embedding function using LangChain's HuggingFaceEmbeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 10: Create a retriever using FAISS and adapt it for use with RetrievalQA
faiss_retriever = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_function
).as_retriever()

# Step 11: Define a function to pass the retrieved documents to an LLM (GPT-3.5 or similar)
def generate_answer_from_documents(query):
    # Load OpenAI API key from environment
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables!")
    
    # Initialize the language model (using OpenAI GPT-3.5)
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-0125")

    # Create a retrieval QA chain with the retriever and LLM
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=faiss_retriever)

    # Debugging: Check FAISS document retrieval
    print(f"Query: {query}")
    docs = faiss_retriever.invoke(query)  # Using the correct method for retrieval
    print(f"Retrieved docs: {docs}")

    # Use the chain to generate the final answer
    answer = rag_chain.invoke({"query": query})
    
    return answer

# Example workflow:
query = "where is Yvonne Farmer lived?"

# Step 12: Pass retrieved documents to the LLM for final answer generation
final_answer = generate_answer_from_documents(query)

# Display the final answer
for key, value in final_answer.items():
    print(f"{key}: {value}")
